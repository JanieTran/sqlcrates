from config import conn, logger


def execute_sql(query: str) -> dict:
    """Run a single DuckDB query and return the result.

    Wraps ``conn.execute()`` in a try/except so the caller never has to handle
    DuckDB errors directly — every failure is captured in the returned dict's
    ``error`` field.

    Parameters
    ----------
    query
        A single DuckDB SQL statement.  Multi-statement strings (separated by
        ``;``) are **not** supported — the caller should split and call this
        function for each one individually.

    Returns
    -------
    dict
        ``{"query": str, "columns": list[str], "rows": list[list], "row_count": int, "error": str | None}``

        When the query succeeds ``error`` is *None*, ``columns`` holds the
        column names returned by the query, and ``rows`` contains the result
        rows as lists of Python values.  When the query fails ``error`` is the
        exception message and both ``columns`` and ``rows`` are empty.
    """
    logger.info("Executing SQL (%d chars):\n%s", len(query), query)
    try:
        # DuckDB's Python API returns a ``DuckDBPyResult`` whose ``description``
        # attribute lists the output columns and ``fetchall()`` materialises
        # every row in one call (acceptable for our 100k-row cap).
        result = conn.execute(query)
        columns = [desc[0] for desc in result.description] if result.description else []
        rows = result.fetchall()
        return {
            "query": query,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "error": None,
        }
    except Exception as exc:
        logger.error("SQL execution failed: %s", exc)
        return {
            "query": query,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "error": str(exc),
        }


def display_query_results(columns: list[str], rows: list[list]) -> None:
    """Log a result table with vertically-aligned columns via a single ``logger.info`` call.

    Each column's width is dynamically computed from the longer of the column
    name and any value in that column, so the output stays readable regardless
    of data length.

    Parameters
    ----------
    columns
        Ordered list of column names.
    rows
        Result rows, each a list of values in the same order as *columns*.
    """
    if not columns or not rows:
        logger.info("  (empty result)")
        return

    widths = [len(c) for c in columns]
    for row in rows:
        for j, val in enumerate(row):
            widths[j] = max(widths[j], len(str(val)))

    lines = []
    header = "  │ ".join(c.ljust(widths[i]) for i, c in enumerate(columns))
    sep = "  ─┼─".join("─" * widths[i] for i in range(len(columns)))
    lines.append(f"  {header}")
    lines.append(f"  {sep}")
    for row in rows:
        line = "  │ ".join(str(val).ljust(widths[j]) for j, val in enumerate(row))
        lines.append(f"  {line}")

    logger.info("\n".join(lines))
