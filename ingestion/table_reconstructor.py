import re

def reconstruct_tables(raw_text: str) -> str:
    """
    Detect linearised tables in plain text and reformat them as
    explicit key:value pairs so the chunker preserves associations.
    """
    lines = raw_text.split("\n")
    output_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect table-like lines: multiple tab/pipe separators or
        # consistent multi-column spacing (>=2 separators per line)
        if _is_table_row(line):
            table_lines = []
            while i < len(lines) and (_is_table_row(lines[i]) or lines[i].strip() == ""):
                if lines[i].strip():
                    table_lines.append(lines[i])
                i += 1

            if len(table_lines) >= 2:
                reconstructed = _format_table(table_lines)
                output_lines.append("TABLE_START")
                output_lines.extend(reconstructed)
                output_lines.append("TABLE_END")
                continue
        else:
            output_lines.append(line)

        i += 1

    return "\n".join(output_lines)

def _is_table_row(line: str) -> bool:
    # Tabs, pipes, or >=3 consecutive spaces as column separators
    return (
        line.count("\t") >= 2
        or line.count("|") >= 2
        or len(re.findall(r"   +", line)) >= 2
    )

def _format_table(lines: list[str]) -> list[str]:
    """Convert detected table rows to Header: Value pairs."""
    # Split each row on tab/pipe/multi-space
    rows = [re.split(r"\t|\|{1,2}|   +", l.strip()) for l in lines]
    rows = [[cell.strip() for cell in row if cell.strip()] for row in rows]

    if not rows:
        return lines

    headers = rows[0]
    result = []
    for data_row in rows[1:]:
        for j, cell in enumerate(data_row):
            if j < len(headers) and cell:
                result.append(f"{headers[j]}: {cell}")
    return result if result else [" | ".join(c for r in rows for c in r)]
