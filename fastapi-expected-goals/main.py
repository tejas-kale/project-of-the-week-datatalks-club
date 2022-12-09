"""
To run this file, execute the following command in the terminal:

uvicorn main:app --reload

Documentation of the app can be viewed at http://127.0.0.1:8000/docs.
"""
from typing import Optional

from fastapi import FastAPI

# Create app instance.
app = FastAPI()


# `"/"` defines the route and denotes the landing page.
@app.get("/")
def read_root():
    # Dictionary is automatically converted to a JSON response.
    return {"Hello": "World"}


"""
- Besides `get`, other supported operators are `put`, `post`, and `delete`.
- The type hint also checks if the provided input matches the specified type. For
`item_id`, if we provide a string input, the server does not crash but sends an error
as follows:

```
{"detail":[{"loc":["path","item_id"],"msg":"value is not a valid integer","type":"type_error.integer"}]}
```
- We can specify the query (`q`) parameter as follows:

```
http://127.0.0.1:8000/items/5?q=test
```

- Any of the functions can be made asynchronous by adding the `async` keyword before function definition.
For instance:

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
   pass
```
"""
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}
