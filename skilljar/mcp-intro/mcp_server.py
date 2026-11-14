from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DocumentMCP")

docs = {
    "deposition.md": "Testimony of Angela Smith, P.E.",
    "plan.md": "Outline of the project implementation steps.",
}

@mcp.tool()
def read_doc(doc_id: str) -> str:
    """Read the contents of a document by its id."""
    if doc_id not in docs:
        raise ValueError(f"Doc {doc_id} not found")
    return docs[doc_id]

@mcp.resource("docs://documents")
def list_docs() -> list[str]:
    """List all document ids."""
    return list(docs.keys())

if __name__ == "__main__":
    mcp.run(transport="stdio")

    