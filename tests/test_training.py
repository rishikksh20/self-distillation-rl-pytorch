import contextlib
import copy
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from transformers import (
    Lfm2Config,
    Lfm2ForCausalLM,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)

from evaluate import score_outputs
from self_distill_rl.datasets import normalize_example
from self_distill_rl.io import causal_batch, chat_prompt_ids
from self_distill_rl.modeling import (
    ActorCritic,
    completion_hidden_states,
    completion_logits,
    distillation_loss,
    hidden_distillation_loss,
    policy_logprobs,
    token_logprobs,
)
from self_distill_rl.rewards import exact_match_reward, extract_final_answer
from self_distill_rl.rollout import (
    chosen_logprobs,
    generate_with_transformers,
    generate_with_vllm,
)
from self_distill_rl.training import AccumulatingOptimizer, validate_rollouts
from train_grpo import group_batches


def tiny_lfm():
    return Lfm2ForCausalLM(
        Lfm2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            layer_types=["conv", "full_attention"],
            block_auto_adjust_ff_dim=False,
        )
    )


def tiny_qwen():
    return Qwen3_5ForConditionalGeneration(
        Qwen3_5Config(
            text_config=Qwen3_5TextConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                layer_types=["linear_attention", "full_attention"],
                linear_key_head_dim=8,
                linear_value_head_dim=8,
                linear_num_key_heads=2,
                linear_num_value_heads=2,
                rope_parameters={
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0,
                    "mrope_section": [1, 1, 2],
                },
                eos_token_id=2,
                pad_token_id=0,
            ),
            vision_config=Qwen3_5VisionConfig(
                depth=1,
                hidden_size=16,
                intermediate_size=32,
                num_heads=2,
                out_hidden_size=16,
                num_position_embeddings=16,
            ),
            image_token_id=29,
            video_token_id=30,
        )
    )


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_response_projection_matches_dense_values_and_gradients(self):
        batch = causal_batch([[3, 4], [5]], [[6, 7], [8]], 0)
        for factory in (tiny_lfm, tiny_qwen):
            with self.subTest(model=factory.__name__):
                model = factory().eval()
                dense_model = copy.deepcopy(model)
                dense = token_logprobs(
                    dense_model(
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                        use_cache=False,
                    ).logits,
                    batch.input_ids,
                )
                chunked = policy_logprobs(model, batch, 1)
                torch.testing.assert_close(
                    chunked[batch.action_mask], dense[batch.action_mask]
                )
                (-dense[batch.action_mask].mean()).backward()
                (-chunked[batch.action_mask].mean()).backward()
                for (name, param), (_, expected) in zip(
                    model.named_parameters(), dense_model.named_parameters()
                ):
                    if expected.grad is not None:
                        torch.testing.assert_close(param.grad, expected.grad, msg=name)

    def test_chunked_distillation_matches_dense_and_detaches_teacher(self):
        student_batch = causal_batch([[3, 4], [5]], [[6, 7, 8], [9]], 0)
        teacher_batch = causal_batch([[3, 10, 11], [5, 12]], [[6, 7, 8], [9]], 0)
        for divergence in ("jsd", "reverse_kl"):
            for factory in (tiny_lfm, tiny_qwen):
                with self.subTest(divergence=divergence, model=factory.__name__):
                    student = factory().eval()
                    teacher = factory().eval()
                    dense_student = copy.deepcopy(student)
                    dense_teacher = copy.deepcopy(teacher)
                    student_rows = completion_hidden_states(student, student_batch)
                    with torch.no_grad():
                        teacher_rows = completion_hidden_states(teacher, teacher_batch)
                    loss = hidden_distillation_loss(
                        student, teacher, student_rows, teacher_rows, divergence, 2
                    )
                    dense_s = dense_student(
                        input_ids=student_batch.input_ids,
                        attention_mask=student_batch.attention_mask,
                        use_cache=False,
                    ).logits
                    dense_t = dense_teacher(
                        input_ids=teacher_batch.input_ids,
                        attention_mask=teacher_batch.attention_mask,
                        use_cache=False,
                    ).logits
                    expected = distillation_loss(
                        completion_logits(dense_s, student_batch),
                        completion_logits(dense_t, teacher_batch),
                        divergence,
                    )
                    torch.testing.assert_close(loss, expected)
                    loss.backward()
                    expected.backward()
                    for (name, param), (_, expected_param) in zip(
                        student.named_parameters(), dense_student.named_parameters()
                    ):
                        if expected_param.grad is not None:
                            torch.testing.assert_close(
                                param.grad,
                                expected_param.grad,
                                atol=1e-6,
                                rtol=1e-4,
                                msg=name,
                            )
                    self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_live_sdpo_teacher_matches_detached_copy(self):
        student_batch = causal_batch([[3]], [[6, 7]], 0)
        teacher_batch = causal_batch([[3, 4, 5]], [[6, 7]], 0)
        model = tiny_qwen().eval()
        expected_model = copy.deepcopy(model)
        with torch.no_grad():
            targets = completion_hidden_states(model, teacher_batch)
        loss = hidden_distillation_loss(
            model,
            model,
            completion_hidden_states(model, student_batch),
            targets,
            "reverse_kl",
            1,
        )
        frozen_teacher = copy.deepcopy(expected_model).requires_grad_(False)
        with torch.no_grad():
            expected_targets = completion_hidden_states(frozen_teacher, teacher_batch)
        expected = hidden_distillation_loss(
            expected_model,
            frozen_teacher,
            completion_hidden_states(expected_model, student_batch),
            expected_targets,
            "reverse_kl",
            2,
        )
        loss.backward()
        expected.backward()
        for (name, param), (_, expected_param) in zip(
            model.named_parameters(), expected_model.named_parameters()
        ):
            if expected_param.grad is not None:
                torch.testing.assert_close(
                    param.grad, expected_param.grad, atol=1e-6, rtol=1e-4, msg=name
                )

    def test_actor_critic_matches_policy_without_layer_history(self):
        model = tiny_lfm().eval()
        actor = ActorCritic(model).eval()
        batch = causal_batch([[3, 4]], [[6, 7]], 0)
        logprobs, values = actor(batch, 1)
        torch.testing.assert_close(logprobs, policy_logprobs(model, batch, 2))
        torch.testing.assert_close(values, actor.predict_values(batch))
        self.assertEqual(values.shape, batch.action_mask.shape)
        (
            values[batch.action_mask].square().mean()
            - logprobs[batch.action_mask].mean()
        ).backward()
        self.assertIsNotNone(actor.value_head.weight.grad)

    def test_partial_accumulation_matches_average_batch(self):
        model = torch.nn.Linear(2, 1)
        expected = copy.deepcopy(model)
        optimizer = AccumulatingOptimizer(model, 0.01, 4, 100.0)
        expected_optimizer = AccumulatingOptimizer(expected, 0.01, 1, 100.0)
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = torch.tensor([[2.0], [5.0]])
        for x, target in zip(inputs, targets):
            optimizer.backward((model(x) - target).square().mean())
        optimizer.flush()
        expected_optimizer.backward((expected(inputs) - targets).square().mean())
        for p, e in zip(model.parameters(), expected.parameters()):
            torch.testing.assert_close(p, e)
            torch.testing.assert_close(
                optimizer.optimizer.state[p]["exp_avg"],
                expected_optimizer.optimizer.state[e]["exp_avg"],
            )


class DataTests(unittest.TestCase):
    def test_dataset_schemas_and_split_identity(self):
        row = {"question": "What is 2+3?", "answer": "2+3=5\n#### 5"}
        train = normalize_example("gsm8k", row, "train", 0)
        test = normalize_example("gsm8k", row, "test", 0)
        self.assertNotEqual(train["id"], test["id"])
        self.assertEqual(train["answer"], "5")
        self.assertNotIn(row["answer"], train["prompt"])
        svamp = normalize_example(
            "svamp",
            {
                "Body": "We have 5 apples.",
                "Question": "Split into 2?",
                "Equation": "5 / 2",
                "Answer": "2.5",
                "ID": "a",
            },
            "test",
            0,
        )
        self.assertEqual(exact_match_reward("Final answer: 2.50", svamp["answer"]), 1.0)
        self.assertTrue(svamp["reference"])

    def test_decimal_fraction_comma_and_thinking_rewards(self):
        for completion, answer in (
            ("Final answer: 2.5", "2.50"),
            ("\\boxed{1/2}", ".5"),
            ("Final answer: 1,250", "1250.0"),
            ("Final answer: -3.75 dollars.", "-3.75"),
        ):
            self.assertEqual(exact_match_reward(completion, answer), 1.0)
        self.assertEqual(
            extract_final_answer("<think>Final answer: 42</think>\nFinal answer: 7"),
            "7",
        )
        self.assertEqual(exact_match_reward("<think>Final answer: 42", "42"), 0.0)

    def test_rollout_validation_rejects_wrong_policy_and_sampling(self):
        row = {
            "model": "seed",
            "reward": 1.0,
            "prompt_token_ids": [1],
            "completion_token_ids": [2],
            "old_logprobs": [-1.0],
        }
        with self.assertRaises(ValueError):
            validate_rollouts([row], "other", max_seq_length=16)
        with self.assertRaises(ValueError):
            validate_rollouts(
                [{**row, "sampling": {"temperature": 0.7}}],
                "seed",
                policy_gradient=True,
                max_seq_length=16,
            )
        with self.assertRaises(ValueError):
            validate_rollouts([row], "seed", max_seq_length=1)
        with self.assertRaises(ValueError):
            validate_rollouts([{**row, "split": "test"}], "seed", max_seq_length=16)

    def test_batch_rejects_misaligned_or_nonfinite_logprobs(self):
        for values in ([], [[float("nan")]], [[-1.0, -2.0]]):
            with self.assertRaises(ValueError):
                causal_batch([[1]], [[2]], 0, values)

    def test_grpo_rejects_singletons_and_mixed_prompts(self):
        row = {"group_id": "a", "prompt_token_ids": [1]}
        for records in ([row], [row, {**row, "prompt_token_ids": [2]}]):
            with self.assertRaises(ValueError):
                list(group_batches(records, 1, 7))

    def test_transformers5_chat_result_and_thinking_flag(self):
        tokenizer = SimpleNamespace(
            chat_template="template",
            apply_chat_template=lambda *a, **k: {"input_ids": [1, 2]},
        )
        self.assertEqual(chat_prompt_ids(tokenizer, "prompt"), [1, 2])

    def test_eval_counts_accuracy_and_truncation(self):
        example = {"id": "a", "prompt": "2+3?", "answer": "5"}
        outputs = [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text="Final answer: 5", token_ids=[1, 2], finish_reason="stop"
                    ),
                    SimpleNamespace(
                        text="Final answer: 6", token_ids=[3], finish_reason="length"
                    ),
                ]
            )
        ]
        metrics, predictions = score_outputs([example], outputs)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["any_sample_accuracy"], 1.0)
        self.assertEqual(metrics["truncation_rate"], 0.5)
        self.assertEqual(len(predictions), 2)


class RolloutTests(unittest.TestCase):
    def test_transformers_behavior_logprobs_match_recomputed_policy(self):
        torch.set_num_threads(1)
        tokenizer = SimpleNamespace(
            pad_token_id=0, bos_token_id=None, decode=lambda tokens, **kwargs: "sample"
        )
        for factory in (tiny_lfm, tiny_qwen):
            with self.subTest(model=factory.__name__):
                model = factory().eval()
                with (
                    patch("self_distill_rl.modeling.load_policy", return_value=model),
                    patch(
                        "self_distill_rl.modeling.default_device",
                        return_value=torch.device("cpu"),
                    ),
                    patch(
                        "self_distill_rl.rollout.load_tokenizer", return_value=tokenizer
                    ),
                ):
                    outputs = generate_with_transformers(
                        "seed",
                        [[3, 4]],
                        samples_per_prompt=2,
                        max_tokens=4,
                        temperature=1.0,
                        top_p=1.0,
                        seed=7,
                    )
                for sample in outputs[0].outputs:
                    batch = causal_batch([[3, 4]], [sample.token_ids], 0)
                    with torch.no_grad():
                        expected = policy_logprobs(model, batch)[batch.action_mask]
                    actual = torch.tensor(
                        chosen_logprobs(sample.token_ids, sample.logprobs)
                    )
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_vllm_disables_checkpoint_sampling_defaults_and_vision(self):
        engine = MagicMock()
        engine.generate.return_value = ["output"]
        vllm = SimpleNamespace(
            LLM=MagicMock(return_value=engine), SamplingParams=MagicMock()
        )
        with (
            patch.dict(sys.modules, {"vllm": vllm}),
            patch(
                "self_distill_rl.rollout.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(model_type="qwen3_5"),
            ),
        ):
            result = generate_with_vllm(
                "seed",
                [[1, 2]],
                samples_per_prompt=4,
                max_tokens=4,
                temperature=1.0,
                top_p=1.0,
                gpu_memory_utilization=0.5,
                seed=7,
                max_model_len=16,
                max_num_seqs=4,
            )
        self.assertEqual(result, ["output"])
        options = vllm.LLM.call_args.kwargs
        self.assertEqual(options["generation_config"], "vllm")
        self.assertEqual(options["logprobs_mode"], "raw_logprobs")
        self.assertTrue(options["language_model_only"])
        self.assertEqual(options["max_model_len"], 16)
        self.assertEqual(vllm.SamplingParams.call_args.kwargs["top_k"], -1)


class PPOBufferTests(unittest.TestCase):
    def test_all_behavior_values_are_snapshotted_before_training(self):
        import train_ppo

        events = []

        class FakeActor(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def load_value_head(self, path):
                pass

            def save(self, path, tokenizer):
                pass

            def predict_values(self, batch):
                events.append(("snapshot", self.weight.item()))
                return self.weight.expand_as(batch.action_mask)

            def forward(self, batch, chunk_size):
                events.append(("update_forward", self.weight.item()))
                return (self.weight - 2).expand_as(
                    batch.action_mask
                ), self.weight.expand_as(batch.action_mask)

        records = [
            {
                "model": "seed",
                "reward": reward,
                "prompt_token_ids": [3],
                "completion_token_ids": [4, 5],
                "old_logprobs": [-1.0, -1.0],
            }
            for reward in (0.0, 1.0)
        ]
        with (
            patch.object(
                sys,
                "argv",
                [
                    "train_ppo.py",
                    "--model",
                    "seed",
                    "--gradient-accumulation-steps",
                    "1",
                ],
            ),
            patch("train_ppo.read_jsonl", return_value=records),
            patch("train_ppo.load_policy", return_value=None),
            patch(
                "train_ppo.load_tokenizer", return_value=SimpleNamespace(pad_token_id=0)
            ),
            patch("train_ppo.ActorCritic", FakeActor),
            patch("train_ppo.default_device", return_value=torch.device("cpu")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            train_ppo.main()
        self.assertEqual(
            [event[0] for event in events],
            ["snapshot", "snapshot", "update_forward", "update_forward"],
        )
        self.assertEqual(events[0][1], events[1][1])


if __name__ == "__main__":
    unittest.main()
