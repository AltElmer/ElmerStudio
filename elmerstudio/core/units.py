"""Unit-aware expression evaluation in the COMSOL style: ``10[mm]``, ``20[degC]``, ``k0*(1+a*(T-T0))``.

Every value is converted to SI on evaluation. Expressions may reference parameters,
math functions and (for SIF export) field variables, which are translated to MATC.
"""
from __future__ import annotations

import ast
import math
import re
from functools import lru_cache

import numpy as np


class ExprError(ValueError):
    pass


# --------------------------------------------------------------------------- units
_PREFIX = {"Y": 1e24, "Z": 1e21, "E": 1e18, "P": 1e15, "T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3,
           "h": 1e2, "da": 1e1, "d": 1e-1, "c": 1e-2, "m": 1e-3, "u": 1e-6, "µ": 1e-6,
           "n": 1e-9, "p": 1e-12, "f": 1e-15}
# base units that accept SI prefixes
_PREFIXABLE = {"m": 1.0, "g": 1e-3, "s": 1.0, "A": 1.0, "K": 1.0, "mol": 1.0, "cd": 1.0, "N": 1.0,
               "Pa": 1.0, "J": 1.0, "W": 1.0, "C": 1.0, "V": 1.0, "F": 1.0, "ohm": 1.0, "S": 1.0,
               "Wb": 1.0, "T": 1.0, "H": 1.0, "Hz": 1.0, "L": 1e-3, "l": 1e-3, "eV": 1.602176634e-19,
               "bar": 1e5, "Ohm": 1.0, "Ω": 1.0}
_PLAIN = {"1": 1.0, "rad": 1.0, "sr": 1.0, "deg": math.pi / 180.0, "min": 60.0, "h": 3600.0, "d": 86400.0,
          "day": 86400.0, "yr": 365.25 * 86400.0, "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344,
          "lb": 0.45359237, "lbf": 4.4482216152605, "psi": 6894.757293168, "atm": 101325.0,
          "torr": 133.322368, "mmHg": 133.322387415, "kWh": 3.6e6, "Wh": 3600.0, "cal": 4.184,
          "kcal": 4184.0, "rpm": 2 * math.pi / 60.0, "percent": 0.01, "ppm": 1e-6, "gal": 3.785411784e-3,
          "erg": 1e-7, "dyn": 1e-5, "G": 1e-4, "Oe": 1000.0 / (4 * math.pi), "liter": 1e-3}
# affine temperature units: SI = factor*value + offset
_AFFINE = {"degC": (1.0, 273.15), "°C": (1.0, 273.15), "degF": (5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
           "°F": (5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0), "degR": (5.0 / 9.0, 0.0)}


def _atom_factor(sym: str) -> float:
    if sym in _PLAIN:
        return _PLAIN[sym]
    if sym in _PREFIXABLE:
        return _PREFIXABLE[sym]
    if sym in _AFFINE:
        return _AFFINE[sym][0]
    for p in sorted(_PREFIX, key=len, reverse=True):
        if sym.startswith(p) and sym[len(p):] in _PREFIXABLE:
            return _PREFIX[p] * _PREFIXABLE[sym[len(p):]]
    raise ExprError(f"Unknown unit '{sym}'")


_UNIT_TOKEN = re.compile(r"[A-Za-zµΩ°_][A-Za-z0-9µΩ°_]*")


@lru_cache(maxsize=1024)
def unit_info(unit: str) -> tuple[float, float]:
    """Return (factor, offset) converting a value in *unit* to SI: si = factor*v + offset."""
    u = unit.strip()
    if u in ("", "1"):
        return 1.0, 0.0
    if u in _AFFINE:
        return _AFFINE[u]
    expr = u.replace("·", "*").replace("^", "**").replace(" ", "*")
    names = {}

    def repl(m):
        tok = m.group(0)
        key = f"__u{len(names)}"
        names[key] = _atom_factor(tok)
        return key

    py = _UNIT_TOKEN.sub(repl, expr)
    try:
        val = _safe_eval(ast.parse(py, mode="eval").body, names, {})
    except ExprError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise ExprError(f"Invalid unit '{unit}': {exc}") from exc
    return float(val), 0.0


def to_si(value: float, unit: str) -> float:
    f, o = unit_info(unit)
    return value * f + o


def from_si(value, unit: str):
    f, o = unit_info(unit)
    return (value - o) / f


def is_valid_unit(unit: str) -> bool:
    try:
        unit_info(unit)
        return True
    except ExprError:
        return False


# --------------------------------------------------------------------------- expressions
MATH_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "atan2": np.arctan2, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "exp": np.exp, "log": np.log,
    "log10": np.log10, "sqrt": np.sqrt, "abs": np.abs, "min": np.minimum, "max": np.maximum,
    "sign": np.sign, "floor": np.floor, "ceil": np.ceil, "round": np.round, "pow": np.power,
}
CONSTANTS = {"pi": math.pi, "e_const": math.e, "g_const": 9.80665, "epsilon0_const": 8.8541878128e-12,
             "mu0_const": 1.25663706212e-6, "sigma_const": 5.670374419e-8, "R_const": 8.314462618,
             "k_B_const": 1.380649e-23, "N_A_const": 6.02214076e23, "F_const": 96485.33212,
             "c_const": 299792458.0, "h_const": 6.62607015e-34, "e": math.e, "i": 1j}

# MATC function names for SIF export
_MATC_FUNCS = {"sin": "sin", "cos": "cos", "tan": "tan", "asin": "asin", "acos": "acos", "atan": "atan",
               "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "exp": "exp", "log": "ln", "log10": "log",
               "sqrt": "sqrt", "abs": "abs", "floor": "floor", "ceil": "ceil", "pow": None, "min": "min",
               "max": "max"}

_BRACKET = re.compile(r"\[([^\[\]]*)\]")
_NUM_AFFINE = re.compile(r"(\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)\s*\[\s*(degC|degF|°C|°F)\s*\]")


def preprocess(expr: str) -> str:
    """COMSOL syntax -> Python syntax (units folded into numeric factors)."""
    s = str(expr).strip()
    if not s:
        raise ExprError("Empty expression")

    def affine(m):
        v = float(m.group(1))
        return repr(to_si(v, m.group(2)))

    s = _NUM_AFFINE.sub(affine, s)

    def unit(m):
        f, o = unit_info(m.group(1))
        if o:
            raise ExprError(f"Affine unit [{m.group(1)}] must follow a number")
        return f"*{f!r}"

    s = _BRACKET.sub(unit, s)
    s = s.replace("^", "**")
    return s


_BIN = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b, ast.Mod: lambda a, b: a % b}
_CMP = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal,
        ast.Eq: np.equal, ast.NotEq: np.not_equal}


def _safe_eval(node, names: dict, funcs: dict):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ExprError(f"Unsupported constant {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise ExprError(f"Unknown symbol '{node.id}'")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_safe_eval(node.left, names, funcs), _safe_eval(node.right, names, funcs))
    if isinstance(node, ast.UnaryOp):
        v = _safe_eval(node.operand, names, funcs)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        if isinstance(node.op, ast.Not):
            return np.logical_not(v)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMP:
        a = _safe_eval(node.left, names, funcs)
        b = _safe_eval(node.comparators[0], names, funcs)
        return _CMP[type(node.ops[0])](a, b).astype(float) if isinstance(a, np.ndarray) else float(_CMP[type(node.ops[0])](a, b))
    if isinstance(node, ast.BoolOp):
        vals = [_safe_eval(v, names, funcs) for v in node.values]
        out = vals[0]
        for v in vals[1:]:
            out = np.logical_and(out, v) if isinstance(node.op, ast.And) else np.logical_or(out, v)
        return out * 1.0
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = funcs.get(node.func.id) or MATH_FUNCS.get(node.func.id)
        if fn is None:
            raise ExprError(f"Unknown function '{node.func.id}'")
        args = [_safe_eval(a, names, funcs) for a in node.args]
        return fn(*args)
    if isinstance(node, ast.IfExp):
        c = _safe_eval(node.test, names, funcs)
        return np.where(c, _safe_eval(node.body, names, funcs), _safe_eval(node.orelse, names, funcs))
    raise ExprError(f"Unsupported syntax: {ast.dump(node)[:60]}")


@lru_cache(maxsize=4096)
def _parse(expr: str):
    py = preprocess(expr)
    try:
        return ast.parse(py, mode="eval").body
    except SyntaxError as exc:
        raise ExprError(f"Syntax error in '{expr}'") from exc


def free_symbols(expr: str) -> set[str]:
    tree = _parse(str(expr))
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id not in CONSTANTS:
            out.add(n.id)
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            out.discard(n.func.id)
    return out


def evaluate(expr, names: dict | None = None, funcs: dict | None = None):
    """Evaluate *expr* (str or number) to SI. Arrays are allowed in *names*."""
    if isinstance(expr, (int, float, np.ndarray)):
        return expr
    tree = _parse(str(expr))
    val = _safe_eval(tree, names or {}, funcs or {})
    if isinstance(val, complex) and val.imag == 0:
        val = val.real
    return val


def try_evaluate(expr, names=None, default=None):
    try:
        return evaluate(expr, names)
    except Exception:
        return default


def evaluate_parameters(rows: list[tuple[str, str]], overrides: dict | None = None) -> tuple[dict, dict]:
    """Evaluate a parameter table in order. Returns (values, errors)."""
    vals: dict[str, float] = {}
    errs: dict[str, str] = {}
    overrides = overrides or {}
    pending = list(rows)
    # allow forward references by iterating until no progress
    for _ in range(len(pending) + 1):
        nxt = []
        for name, expr in pending:
            if name in overrides:
                vals[name] = overrides[name]
                continue
            try:
                vals[name] = float(np.real(evaluate(expr, vals)))
                errs.pop(name, None)
            except ExprError as exc:
                errs[name] = str(exc)
                nxt.append((name, expr))
        if len(nxt) == len(pending):
            break
        pending = nxt
    return vals, errs


# --------------------------------------------------------------------------- MATC export
def _fmt(v: float) -> str:
    if isinstance(v, complex):
        v = v.real
    if v == int(v) and abs(v) < 1e15:
        return f"{int(v)}"
    return f"{v:.12g}"


def to_matc(expr: str, params: dict, fields: dict[str, str]):
    """Translate an expression to an Elmer SIF value.

    Returns ``float`` when constant, else ``(variables, matc_string)`` where
    *variables* is the list of Elmer variable names the MATC code depends on.
    ``fields`` maps expression symbols (e.g. ``T``) to Elmer variables (``Temperature``).
    """
    tree = _parse(str(expr))
    used = [s for s in sorted(free_symbols(expr)) if s not in params]
    unknown = [s for s in used if s not in fields]
    if unknown:
        raise ExprError(f"Unknown symbol(s) {', '.join(unknown)} in '{expr}'")
    if not used:
        val = evaluate(expr, params)
        return float(np.real(val))
    var_index = {s: i for i, s in enumerate(used)}
    single = len(used) == 1

    def emit(n) -> str:
        if isinstance(n, ast.Constant):
            return _fmt(n.value)
        if isinstance(n, ast.Name):
            if n.id in var_index:
                return "tx" if single else f"tx({var_index[n.id]})"
            if n.id in params:
                return _fmt(params[n.id])
            if n.id in CONSTANTS:
                return _fmt(CONSTANTS[n.id])
            raise ExprError(f"Unknown symbol '{n.id}'")
        if isinstance(n, ast.BinOp):
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "^"}.get(type(n.op))
            if op is None:
                raise ExprError("Operator not supported in MATC export")
            return f"({emit(n.left)}{op}{emit(n.right)})"
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return ("-" if isinstance(n.op, ast.USub) else "") + f"({emit(n.operand)})"
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            name = n.func.id
            args = [emit(a) for a in n.args]
            if name == "pow":
                return f"(({args[0]})^({args[1]}))"
            if name not in _MATC_FUNCS:
                raise ExprError(f"Function '{name}' cannot be exported to MATC")
            return f"{_MATC_FUNCS[name]}({', '.join(args)})"
        raise ExprError("Expression not exportable to MATC")

    code = emit(tree)
    return [fields[s] for s in used], code
