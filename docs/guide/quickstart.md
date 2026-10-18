# Quickstart

!!! note
    Stub page (E0-T04). Filled in once the `Project` facade and CLI (E8-T02) exist; completed in E8-T04.

The intended end-to-end flow (see E8-T02's `Project` facade):

```python
from regimekit import Project

p = Project.from_yaml("dax_3state.yaml")
data = p.fetch()
fit = p.fit(data)
post = p.decode(fit, data)
fc = p.forecast(fit, data, horizon=20)
```
