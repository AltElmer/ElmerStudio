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
    "real": np.real, "imag": np.imag, "arg": np.angle, "conj": np.conj,
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


_OPERAND = re.compile(r"(?:\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|[A-Za-z_][\w.]*)$")


def _attach_units(s: str, factor) -> str:
    """Bind every [unit] to the operand right before it, as COMSOL does: ``x/2[mm]`` is x/(2[mm]), not (x/2)*mm.
    *factor(unit)* returns the Python text the unit becomes (a number, or a name)."""
    out = s
    while True:
        m = _BRACKET.search(out)
        if m is None:
            return out
        head, unit, tail = out[:m.start()].rstrip(), m.group(1), out[m.end():]
        if head.endswith(")"):                         # a parenthesized operand, or a function call
            depth, i = 0, len(head) - 1
            while i >= 0:
                depth += head[i] == ")"
                depth -= head[i] == "("
                if depth == 0:
                    break
                i -= 1
            j = i
            while j > 0 and (head[j - 1].isalnum() or head[j - 1] in "_."):
                j -= 1
            start = j
        else:
            o = _OPERAND.search(head)
            start = o.start() if o else len(head)
        operand = head[start:]
        rep = f"({operand}*{factor(unit)})" if operand else f"{factor(unit)}"
        out = head[:start] + rep + tail


def preprocess(expr: str) -> str:
    """COMSOL syntax -> Python syntax (units folded into numeric factors bound to their operand)."""
    s = str(expr).strip()
    if not s:
        raise ExprError("Empty expression")

    def affine(m):
        v = float(m.group(1))
        return repr(to_si(v, m.group(2)))

    s = _NUM_AFFINE.sub(affine, s)

    def unit(u):
        f, o = unit_info(u)
        if o:
            raise ExprError(f"Affine unit [{u}] must follow a number")
        return repr(f)

    s = _attach_units(s, unit)
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
    # name the dependency that blocks each unresolved parameter: a circular reference, or a missing name
    left = {name for name in errs}
    for name, expr in rows:
        if name not in errs:
            continue
        try:
            missing = sorted(free_symbols(expr) - set(vals))
        except Exception:
            continue
        cyc = _cycle_from(name, rows, left)
        if cyc:
            errs[name] = "Circular reference: " + " -> ".join(cyc)
        elif missing:
            errs[name] = f"'{expr}' depends on undefined {', '.join(missing)}"
    return vals, errs


def _cycle_from(start, rows, unresolved):
    """A dependency cycle through *start* among the unresolved parameters, as a list of names, or None."""
    deps = {}
    for name, expr in rows:
        try:
            deps[name] = free_symbols(expr) & unresolved
        except Exception:
            deps[name] = set()
    path, seen = [start], set()

    def walk(n):
        for d in sorted(deps.get(n, ())):
            if d == start:
                return path + [start]
            if d not in seen:
                seen.add(d)
                path.append(d)
                r = walk(d)
                if r:
                    return r
                path.pop()
        return None
    return walk(start)


# --------------------------------------------------------------------------- dimensions
# Exponents of the SI base quantities (length, mass, time, current, temperature, amount, luminous intensity).
# Expressions are checked like COMSOL does: a value whose deduced unit differs from the field's unit is still used
# (its SI number), and the field says "deduced [..], expected [..]". A plain number carries no unit and is taken in
# the field's unit; adding quantities of different dimensions is reported too.
from fractions import Fraction as _Fr  # noqa: E402


class Dim(tuple):
    """Dimension: exponents of (m, kg, s, A, K, mol, cd)."""

    def __new__(cls, *e):
        e = tuple(_Fr(x).limit_denominator(12) for x in (e or (0,) * 7))
        return super().__new__(cls, e + (_Fr(0),) * (7 - len(e)))

    def __mul__(self, o):
        return Dim(*(a + b for a, b in zip(self, o))) if isinstance(o, Dim) else self

    __rmul__ = __mul__

    def __truediv__(self, o):
        return Dim(*(a - b for a, b in zip(self, o))) if isinstance(o, Dim) else self

    def __rtruediv__(self, o):
        return Dim(*(-a for a in self))

    def __pow__(self, p):
        return Dim(*(a * _Fr(p).limit_denominator(12) for a in self))

    @property
    def none(self) -> bool:
        return not any(self)


DIMLESS = Dim()
_L, _M, _T, _I, _K, _N, _J = (Dim(*([0] * i + [1])) for i in range(7))
_NEWTON = _M * _L / _T ** 2
_JOULE = _NEWTON * _L
_WATT = _JOULE / _T
_PASCAL = _NEWTON / _L ** 2
_VOLT = _WATT / _I
_OHM = _VOLT / _I
_TESLA = _VOLT * _T / _L ** 2
_UNIT_DIMS = {"m": _L, "g": _M, "s": _T, "A": _I, "K": _K, "mol": _N, "cd": _J, "N": _NEWTON, "Pa": _PASCAL,
              "J": _JOULE, "W": _WATT, "C": _I * _T, "V": _VOLT, "F": _I * _T / _VOLT, "ohm": _OHM, "Ohm": _OHM,
              "Ω": _OHM, "S": DIMLESS / _OHM, "Wb": _VOLT * _T, "T": _TESLA, "H": _VOLT * _T / _I, "Hz": DIMLESS / _T,
              "L": _L ** 3, "l": _L ** 3, "eV": _JOULE, "bar": _PASCAL,
              "1": DIMLESS, "rad": DIMLESS, "sr": DIMLESS, "deg": DIMLESS, "min": _T, "h": _T, "d": _T, "day": _T,
              "yr": _T, "in": _L, "ft": _L, "yd": _L, "mi": _L, "lb": _M, "lbf": _NEWTON, "psi": _PASCAL,
              "atm": _PASCAL, "torr": _PASCAL, "mmHg": _PASCAL, "kWh": _JOULE, "Wh": _JOULE, "cal": _JOULE,
              "kcal": _JOULE, "rpm": DIMLESS / _T, "percent": DIMLESS, "ppm": DIMLESS, "gal": _L ** 3, "erg": _JOULE,
              "dyn": _NEWTON, "G": _TESLA, "Oe": _I / _L, "liter": _L ** 3,
              "degC": _K, "°C": _K, "degF": _K, "°F": _K, "degR": _K}
_CONST_DIMS = {"pi": DIMLESS, "e_const": DIMLESS, "e": DIMLESS, "i": DIMLESS, "g_const": _L / _T ** 2,
               "epsilon0_const": _UNIT_DIMS["F"] / _L, "mu0_const": _UNIT_DIMS["H"] / _L,
               "sigma_const": _WATT / (_L ** 2 * _K ** 4), "R_const": _JOULE / (_N * _K), "k_B_const": _JOULE / _K,
               "N_A_const": DIMLESS / _N, "F_const": _I * _T / _N, "c_const": _L / _T, "h_const": _JOULE * _T}
# named units used to print a deduced dimension (first match wins)
_NAMED = ["1", "m", "kg", "s", "A", "K", "mol", "N", "Pa", "J", "W", "C", "V", "ohm", "S", "T", "Wb", "H", "F", "Hz",
          "m^2", "m^3", "m/s", "m/s^2", "kg/m^3", "W/(m*K)", "W/m^2", "W/m^3", "W/(m^2*K)", "J/(kg*K)", "S/m",
          "A/m^2", "V/m", "mol/m^3", "m^2/s", "mol/(m^2*s)", "Pa*s", "1/K", "C/m^3"]


def _atom_dims(sym: str) -> Dim:
    if sym in _UNIT_DIMS:
        return _UNIT_DIMS[sym]
    for p in sorted(_PREFIX, key=len, reverse=True):
        if sym.startswith(p) and sym[len(p):] in _PREFIXABLE:
            base = sym[len(p):]
            return _M if base == "g" else _UNIT_DIMS[base]
    raise ExprError(f"Unknown unit '{sym}'")


@lru_cache(maxsize=1024)
def unit_dims(unit: str) -> Dim:
    """The dimension of a unit expression ("W/(m^2*K)", "mm", "degC", "1")."""
    u = str(unit).strip()
    if u in ("", "1"):
        return DIMLESS
    if u == "LEN":
        return _L
    if u in _UNIT_DIMS:
        return _UNIT_DIMS[u]
    names = {}

    def repl(m):
        key = f"__u{len(names)}"
        names[key] = _atom_dims(m.group(0))
        return key
    py = _UNIT_TOKEN.sub(repl, u.replace("·", "*").replace("^", "**").replace(" ", "*"))
    try:
        d = _dim_eval(ast.parse(py, mode="eval").body, names)
    except ExprError:
        raise
    except Exception as exc:
        raise ExprError(f"Invalid unit '{unit}': {exc}") from exc
    return d if isinstance(d, Dim) else DIMLESS


def _dim_eval(node, names):
    """Dimension arithmetic for unit strings (numbers are exponents or dimensionless factors)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return names[node.id]
    if isinstance(node, ast.BinOp):
        a, b = _dim_eval(node.left, names), _dim_eval(node.right, names)
        if isinstance(node.op, ast.Pow):
            return a ** b if isinstance(a, Dim) else a ** b
        if isinstance(node.op, ast.Mult):
            return a * b if isinstance(a, Dim) or isinstance(b, Dim) else a * b
        if isinstance(node.op, ast.Div):
            if isinstance(a, Dim) or isinstance(b, Dim):
                return (a if isinstance(a, Dim) else DIMLESS) / (b if isinstance(b, Dim) else DIMLESS)
            return a / b
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_dim_eval(node.operand, names)
    raise ExprError("unsupported unit syntax")


def fmt_dim(d: Dim | None) -> str:
    if d is None:
        return "?"
    for name in _NAMED:
        try:
            if unit_dims(name.replace("kg", "kg")) == d:
                return name
        except ExprError:
            continue
    num = [(b, e) for b, e in zip(("m", "kg", "s", "A", "K", "mol", "cd"), d) if e > 0]
    den = [(b, -e) for b, e in zip(("m", "kg", "s", "A", "K", "mol", "cd"), d) if e < 0]

    def part(items):
        return "*".join(b if e == 1 else f"{b}^{e}" for b, e in items)
    s = part(num) or "1"
    if den:
        s += "/" + (part(den) if len(den) == 1 and den[0][1] == 1 else f"({part(den)})")
    return s


class _Q:
    """A deduced quantity: its dimension (None: unknown) and whether it came from plain numbers only."""
    __slots__ = ("dim", "pure", "value")

    def __init__(self, dim, pure=False, value=None):
        self.dim, self.pure, self.value = dim, pure, value


_DIMLESS_FUNCS = {"sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "exp", "log", "log10"}
_SAME_FUNCS = {"abs", "sign", "floor", "ceil", "round", "real", "imag", "conj"}


def expr_dims(expr, symbol_dims: dict | None = None, funcs: dict | None = None):
    """Deduce the dimension of *expr*. Returns (Dim or None when unknown, plain-number flag, warnings).

    *symbol_dims*: {name: Dim or None (unknown) or (Dim, pure)}; unknown names give an unknown result (no
    warning). *funcs*: {name: callable(arg_qs) -> Dim or None} for operators such as coupling operators."""
    symbol_dims = symbol_dims or {}
    warnings: list[str] = []
    s = str(expr).strip()
    if not s:
        raise ExprError("Empty expression")
    names: dict[str, _Q] = {}

    def affine(m):
        key = f"__q{len(names)}"
        names[key] = _Q(_K)
        return key
    s = _NUM_AFFINE.sub(affine, s)

    def unit(u):
        key = f"__q{len(names)}"
        names[key] = _Q(unit_dims(u))
        return key
    s = _attach_units(s, unit).replace("^", "**")
    try:
        tree = ast.parse(s, mode="eval").body
    except SyntaxError as exc:
        raise ExprError(f"Syntax error in '{expr}'") from exc

    def same(a: _Q, b: _Q, what: str) -> _Q:
        if a.dim is None or b.dim is None:
            return _Q(None)
        if a.pure:
            return _Q(b.dim, b.pure)
        if b.pure or a.dim == b.dim:
            return _Q(a.dim, a.pure)
        warnings.append(f"inconsistent units in {what}: [{fmt_dim(a.dim)}] and [{fmt_dim(b.dim)}]")
        return _Q(a.dim)

    def num(n):
        try:
            return float(np.real(_safe_eval(n, {}, {})))
        except Exception:
            return None

    def walk(n) -> _Q:
        if isinstance(n, ast.Constant):
            return _Q(DIMLESS, True, n.value)
        if isinstance(n, ast.Name):
            if n.id in names:
                return names[n.id]
            if n.id in symbol_dims:
                v = symbol_dims[n.id]
                if isinstance(v, tuple) and len(v) == 2 and not isinstance(v, Dim):
                    return _Q(v[0], bool(v[1]))
                return _Q(v)
            if n.id in _CONST_DIMS:
                return _Q(_CONST_DIMS[n.id])
            return _Q(None)
        if isinstance(n, ast.BinOp):
            a, b = walk(n.left), walk(n.right)
            if isinstance(n.op, (ast.Add, ast.Sub, ast.Mod)):
                return same(a, b, "a sum")
            if isinstance(n.op, ast.Mult):
                return _Q(None if a.dim is None or b.dim is None else a.dim * b.dim, a.pure and b.pure)
            if isinstance(n.op, ast.Div):
                return _Q(None if a.dim is None or b.dim is None else a.dim / b.dim, a.pure and b.pure)
            if isinstance(n.op, ast.Pow):
                if b.dim is not None and not b.dim.none:
                    warnings.append(f"an exponent with a unit [{fmt_dim(b.dim)}]")
                if a.dim is None:
                    return _Q(None)
                if a.dim.none:
                    return _Q(DIMLESS, a.pure)
                p = num(n.right)
                if p is None:
                    warnings.append(f"a variable exponent of a quantity with a unit [{fmt_dim(a.dim)}]")
                    return _Q(None)
                return _Q(a.dim ** p, a.pure)
        if isinstance(n, ast.UnaryOp):
            return walk(n.operand)
        if isinstance(n, ast.Compare) and len(n.ops) == 1:
            same(walk(n.left), walk(n.comparators[0]), "a comparison")
            return _Q(DIMLESS, True)
        if isinstance(n, ast.BoolOp):
            for v in n.values:
                walk(v)
            return _Q(DIMLESS, True)
        if isinstance(n, ast.IfExp):
            walk(n.test)
            return same(walk(n.body), walk(n.orelse), "the branches of a condition")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            f = n.func.id
            args = [walk(a) for a in n.args]
            if funcs and f in funcs:
                return _Q(funcs[f](args))
            if f in _DIMLESS_FUNCS:
                for a in args:
                    if a.dim is not None and not a.dim.none and not a.pure:
                        warnings.append(f"{f}() of a quantity with a unit [{fmt_dim(a.dim)}]")
                return _Q(DIMLESS, all(a.pure for a in args))
            if f == "sqrt" and args:
                return _Q(None if args[0].dim is None else args[0].dim ** 0.5, args[0].pure)
            if f in _SAME_FUNCS and args:
                return args[0]
            if f in ("min", "max") and len(args) == 2:
                return same(args[0], args[1], f"{f}()")
            if f == "atan2" and len(args) == 2:
                same(args[0], args[1], "atan2()")
                return _Q(DIMLESS)
            if f == "pow" and len(args) == 2:
                p = num(n.args[1])
                return _Q(None if args[0].dim is None or p is None else args[0].dim ** p, args[0].pure)
            return _Q(None)                       # a user function: its unit is not known
        return _Q(None)

    q = walk(tree)
    return q.dim, q.pure, warnings


def parameter_dims(rows) -> dict:
    """{parameter: (Dim or None, plain-number flag)} over the dependency graph (any row order)."""
    out: dict = {}
    pending = list(rows)
    for _ in range(len(pending) + 1):
        nxt = []
        for name, e in pending:
            try:
                deps = free_symbols(e)
            except Exception:
                out[name] = (None, False)
                continue
            if any(d in dict(pending) and d not in out for d in deps if d != name):
                nxt.append((name, e))
                continue
            try:
                d, pure, _w = expr_dims(e, out)
            except ExprError:
                d, pure = None, False
            out[name] = (d, pure)
        if len(nxt) == len(pending):
            for name, _e in nxt:
                out.setdefault(name, (None, False))
            break
        pending = nxt
    return out


def unit_warnings(expr, expected_unit: str | None, symbol_dims: dict | None = None, funcs=None) -> list[str]:
    """COMSOL-style unit messages for a field: inconsistent sums, and a deduced unit that is not the field's
    (a plain number is taken in the field's unit). The value is used in any case."""
    try:
        d, pure, warns = expr_dims(expr, symbol_dims, funcs)
    except ExprError:
        return []
    if expected_unit is None or d is None or pure:
        return warns
    try:
        exp = unit_dims(expected_unit)
    except ExprError:
        return warns
    if d != exp:
        warns = warns + [f"unit [{fmt_dim(d)}] deduced, [{'m' if expected_unit == 'LEN' else expected_unit}] "
                         "expected (the SI value is used as it is)"]
    return warns


# --------------------------------------------------------------------------- MATC export
def _fmt(v: float) -> str:
    if isinstance(v, complex):
        v = v.real
    if v == int(v) and abs(v) < 1e15:
        return f"{int(v)}"
    return f"{v:.12g}"


def parameter_unit_flags(rows) -> dict[str, bool]:
    """Which parameters carry units, directly ("2[mm]") or through any parameter they depend on.  Resolved over
    the dependency graph, independent of row order (a parameter may refer to one defined further down); a
    circular definition carries no units (and evaluate_parameters reports it)."""
    deps: dict[str, set] = {}
    flags: dict[str, bool] = {}
    for name, e in rows:
        e = str(e)
        flags[name] = "[" in e
        try:
            deps[name] = free_symbols(e)
        except Exception:
            deps[name] = set()
    changed = True
    while changed:                                     # monotone: a flag only turns on, so this terminates
        changed = False
        for name, d in deps.items():
            if not flags[name] and any(flags.get(t, False) for t in d):
                flags[name] = changed = True
    return flags


def expr_has_units(expr, rows=None, flags: dict | None = None) -> bool:
    """Does *expr* carry units, directly ("2[mm]") or through a parameter defined with units?  *rows*: the
    model's parameter rows [(name, expression)] (any order), or precomputed *flags* (parameter_unit_flags)."""
    s = str(expr)
    if "[" in s:
        return True
    if flags is None:
        flags = parameter_unit_flags(rows or [])
    try:
        return any(flags.get(t, False) for t in free_symbols(s))
    except Exception:
        return False


def geometry_value(expr, params, rows, scale):
    """A coordinate in the geometry length unit, COMSOL style: a plain number is in the geometry unit, a value
    with units (or built from parameters with units) is converted from SI.  *scale*: geometry unit in metres."""
    v = float(np.real(evaluate(str(expr), params)))
    return v / scale if expr_has_units(expr, rows) else v


# two-argument min/max for exported expressions (MATC's own min/max reduce one vector).  Defined once at the top
# of a SIF that uses them, so nested clamps do not repeat their arguments.
MATC_PRELUDE = {"stmin(": "$ function stmin(a,b) { _stmin = a*(a<=b) + b*(a>b) }",
                "stmax(": "$ function stmax(a,b) { _stmax = a*(a>=b) + b*(a<b) }"}


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
        if isinstance(n, ast.Compare) and len(n.ops) == 1:      # MATC relational operators give 0/1
            op = {ast.Lt: "<", ast.Gt: ">", ast.LtE: "<=", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "<>"}.get(type(n.ops[0]))
            if op is None:
                raise ExprError("Comparison not supported in MATC export")
            return f"({emit(n.left)}{op}{emit(n.comparators[0])})"
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            name = n.func.id
            args = [emit(a) for a in n.args]
            if name == "pow":
                return f"(({args[0]})^({args[1]}))"
            if name in ("min", "max") and len(args) == 2:      # MATC's min/max are one-argument reductions:
                return f"st{name}({args[0]}, {args[1]})"        # call the two-argument helpers in MATC_PRELUDE
            if name == "sign" and len(args) == 1:
                return f"((({args[0]})>0)-(({args[0]})<0))"
            if name not in _MATC_FUNCS:
                raise ExprError(f"Function '{name}' cannot be exported to MATC")
            return f"{_MATC_FUNCS[name]}({', '.join(args)})"
        raise ExprError("Expression not exportable to MATC")

    code = emit(tree)
    return [fields[s] for s in used], code
