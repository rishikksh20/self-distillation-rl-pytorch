"""A safe arithmetic function tool for veRL's asynchronous agent loop."""

import ast
import operator

from verl.tools.function_tool import function_tool


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_ABS_VALUE = 1_000_000_000_000


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if abs(node.value) > _MAX_ABS_VALUE:
            raise ValueError("number is too large")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("exponent is too large")
        result = _BINARY[type(node.op)](left, right)
        if abs(result) > _MAX_ABS_VALUE:
            raise ValueError("result is too large")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    raise ValueError("only numeric literals and arithmetic operators are allowed")


@function_tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate a short arithmetic expression without Python eval.

    Args:
        expression: Arithmetic using numbers, parentheses, +, -, *, /, //, %, or **.
    """
    if len(expression) > 200:
        raise ValueError("expression is too long")
    tree = ast.parse(expression, mode="eval")
    return str(_evaluate(tree))
