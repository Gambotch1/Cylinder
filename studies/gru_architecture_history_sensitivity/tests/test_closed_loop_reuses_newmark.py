"""Check 11: closed-loop evaluation reuses the existing Newmark
implementation and current handoff protocol. Both evaluate_closed_loop.py
(validation cases, feeds selection) and
evaluate_closed_loop_train_diagnostic.py (training cases, excluded from
selection) delegate to the shared _common.run_coupled_sweep, which is the
one and only place that shells out to coupled_inference.py's own CLI --
neither script, nor the shared function, ever defines/imports its own
integrator or calls run_coupled_viv directly."""
import ast
import inspect

import _common
import evaluate_closed_loop
import evaluate_closed_loop_train_diagnostic


def _calls_in(fn) -> set[str]:
    tree = ast.parse(inspect.getsource(fn))
    return {n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def test_run_coupled_sweep_does_not_reimplement_the_integrator():
    src = inspect.getsource(_common.run_coupled_sweep)
    forbidden = ["newmark", "beta_newmark", "def integrate", "class Integrator"]
    lowered = src.lower()
    for token in forbidden:
        assert token not in lowered, f"found '{token}' -- looks like a reimplemented integrator"


def test_run_coupled_sweep_only_source_of_dynamics_is_the_coupled_inference_subprocess():
    src = inspect.getsource(_common.run_coupled_sweep)
    assert "coupled_inference" in src
    assert "run_coupled_viv" not in _calls_in(_common.run_coupled_sweep), (
        "run_coupled_sweep must not CALL run_coupled_viv directly -- it "
        "should go through coupled_inference.py's CLI, matching "
        "evaluate_all.py's own pattern, so the CFD-loading/warmup/handoff "
        "setup code is never duplicated here.")


def test_neither_evaluate_script_calls_run_coupled_viv_or_defines_its_own_sweep():
    """Both the selection-facing and diagnostic-facing scripts must reuse
    _common.run_coupled_sweep rather than reimplementing the subprocess
    loop or calling run_coupled_viv themselves."""
    for module in (evaluate_closed_loop, evaluate_closed_loop_train_diagnostic):
        src = inspect.getsource(module)
        assert "run_coupled_sweep" in src, f"{module.__name__} must call _common.run_coupled_sweep"
        call_names = {n.func.id for n in ast.walk(ast.parse(src))
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "run_coupled_viv" not in call_names, (
            f"{module.__name__} must not call run_coupled_viv directly")
