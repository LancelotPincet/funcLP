## Structure

Every module or routine lives in a named folder:

```
<name>_LP/
    <name>.py        # main public object
    test_<name>.py   # tests
    _functions/      # private or public helpers
    _doctype.txt     # declares the public structure
```

Modules go in `src/funclp/modules/`.
Routines go in `src/funclp/routines/`.

---

## Object rules

- One file → one main object, same name as the file
- Public names must be unique across the entire package
- Files starting with `_` are private
- Private helpers use relative imports only
- Public objects must be exposed in `funclp/__init__.py`

---

## Public API

Always import from the package root:

```python
# RIGHT
from funclp import MyObject

# WRONG — never import from internal paths
from funclp.modules.myobject_LP.myobject import MyObject
```

Every public object must be registered in `funclp/__init__.py`:

```python
sources = {
    "<name>": "funclp.modules.<name>_LP.<name>",
}
```

This applies to public helpers too, not just the main object.

---

## `_doctype.txt`

One word. Declares which objects in the folder are public.

| Value | Meaning |
|-------|---------|
| `default` | only `<name>` is public |
| `parent` | `<name>` is public; `_functions/` contains public children |
| `collab` | `<name>` and all `_functions/` objects are public |
| `instance` | `<name>` is a public instance; its underlying class is private |

---

## `_functions/`

Contains helpers or related objects for `<name>`.

- One main object per file
- Keep files small and focused
- No unrelated logic here