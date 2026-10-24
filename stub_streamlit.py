"""Entorno simulado de Streamlit/Plotly para EJECUTAR app.py sin las librerías reales (smoke test).
No valida el aspecto visual: valida que ninguna rama del código lance errores de nombres, tipos o datos."""
import sys, types
import pandas as pd


class Ctx:
    """Contenedor (columna, pestaña, formulario): delega los widgets al Streamlit simulado, como hace el real."""
    def __init__(self, owner=None): self._owner = owner
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name):
        if name.startswith("_"): raise AttributeError(name)
        return getattr(self._owner, name) if self._owner is not None else (lambda *a, **k: None)


class Stop(Exception):
    pass


class State(dict):
    pass


class Streamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = State()
        self.secrets = {}
        self.press: set[str] = set()
        self.text: dict[str, str] = {}
        self.calls: list[str] = []
        self.errors: list[str] = []
        self.sidebar = self
        self.components = types.SimpleNamespace(v1=types.SimpleNamespace(html=lambda *a, **k: None))

    # --- widgets con valor de retorno
    def number_input(self, label, *a, **k):
        if "value" in k: return k["value"]
        return a[2] if len(a) > 2 else (a[0] if a else 0)
    def slider(self, label, *a, **k):
        if "value" in k: return k["value"]
        return a[2] if len(a) > 2 else a[0]
    def checkbox(self, label, value=False, **k): return value
    def selectbox(self, label, options, index=0, **k): return list(options)[index] if len(list(options)) else None
    def multiselect(self, label, options, default=None, **k): return list(default) if default is not None else []
    def radio(self, label, options, **k): return list(options)[0]
    def text_input(self, label, *a, **k): return self.text.get(label, "")
    def button(self, label, *a, **k):
        self.calls.append(label); return label in self.press
    def form_submit_button(self, label, *a, **k): return label in self.press
    def file_uploader(self, *a, **k): return None
    # --- contenedores
    def columns(self, spec, **k): return [Ctx(self) for _ in range(spec if isinstance(spec, int) else len(spec))]
    def tabs(self, labels): return [Ctx(self) for _ in labels]
    def form(self, *a, **k): return Ctx(self)
    def spinner(self, *a, **k): return Ctx(self)
    # --- control de flujo / caché
    def error(self, msg='', *a, **k): self.errors.append(str(msg))
    def stop(self): raise Stop()
    def rerun(self): raise Stop()
    def cache_data(self, *a, **k):
        if a and callable(a[0]): return a[0]
        return lambda f: f
    def __getattr__(self, name):                 # markdown, caption, dataframe, metric, plotly_chart, error, ...
        if name.startswith("__"): raise AttributeError(name)
        return lambda *a, **k: None


class Fig:
    def __init__(self, *a, **k): pass
    def add_trace(self, *a, **k): return self
    def update_layout(self, *a, **k): return self


def install():
    st = Streamlit()
    sys.modules["streamlit"] = st
    sys.modules["streamlit.components"] = types.ModuleType("streamlit.components")
    sys.modules["streamlit.components.v1"] = types.ModuleType("streamlit.components.v1")
    sys.modules["streamlit.components.v1"].html = lambda *a, **k: None
    go = types.ModuleType("plotly.graph_objects")
    go.Figure, go.Scatter = Fig, lambda *a, **k: {}
    plotly = types.ModuleType("plotly"); plotly.graph_objects = go
    sys.modules["plotly"], sys.modules["plotly.graph_objects"] = plotly, go
    class _Style:               # DataFrame.style sin jinja2
        def __init__(self, df): self.df = df
        def format(self, *a, **k): return self
    if not hasattr(pd.DataFrame, "_orig_style"):
        pd.DataFrame._orig_style = True
        pd.DataFrame.style = property(lambda self: _Style(self))
    return st
