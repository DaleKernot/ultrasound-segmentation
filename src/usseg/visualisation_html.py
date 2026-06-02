import base64
import pickle
import toml

from usseg.digitized_comparison import digitized_cell_background_style


def generate_html(
    scans,
    annotated_scans,
    digitized_scans,
    tables,
    mean_wave_scans=None,
):
    # Check if the number of scan paths and data tables match
    if len(scans) != len(tables):
        raise ValueError("The number of scan paths and data tables do not match.")
    if mean_wave_scans is None:
        mean_wave_scans = [None] * len(scans)
    if len(mean_wave_scans) != len(scans):
        raise ValueError("mean_wave_scans length must match scans.")

    digitized_value_cols = (
        "Digitized Value (ray)",
        "Digitized Value (morph)",
        "Digitized Value (grow)",
    )

    # Start building the HTML string
    html_str = '<html><head>'
    html_str += '<style>'
    html_str += 'table {'
    # html_str += '    border-collapse: collapse;'
    html_str += '    font-family: Arial, sans-serif;'
    html_str += '    font-size: 10px;'
    html_str += '}'
    html_str += 'table td, table th {'
    html_str += '    border: 1px solid #ddd;'
    html_str += '    padding: 2px;'
    html_str += '    white-space: nowrap;'
    html_str += '}'
    html_str += 'table th {'
    html_str += '    background-color: #f2f2f2;'
    html_str += '    font-weight: bold;'
    html_str += '}'
    html_str += '</style>'
    html_str += '</head><body>'

    html_str += '<div style="overflow-x: scroll;white-space: nowrap;">'
    # Loop over each scan and table and add them to the HTML
    for scan_path, Annotated_scan, Digitized_scan, Mean_wave_scan, table_data in zip(
        scans, annotated_scans, digitized_scans, mean_wave_scans, tables
    ):
        # Add the scan image and processed image to the HTML
        with open(scan_path, 'rb') as f:
            im_b64 = base64.b64encode(f.read()).decode("utf-8")
            html_str += f'<div style="display:inline-block;max-width:100%;padding:10px"><img src="data:image/png;base64,{im_b64}" width="300"></div>'

        if Annotated_scan is not None:
            with open(Annotated_scan, 'rb') as f:
                im_b64 = base64.b64encode(f.read()).decode("utf-8")
                html_str += f'<div style="display:inline-block;max-width:100%";padding:10px><img src="data:image/png;base64,{im_b64}" width="300"></div>'
        else:
            html_str += '<div style="display:inline-block;width:300px></div>'

        if Digitized_scan is not None:
            with open(Digitized_scan, 'rb') as f:
                im_b64 = base64.b64encode(f.read()).decode("utf-8")
                html_str += f'<div style="display:inline-block;max-width:100%";padding:10px><img src="data:image/png;base64,{im_b64}" width="300"></div>'
        else:
            html_str += '<div style="display:inline-block;width:300px"></div>'

        if Mean_wave_scan is not None:
            with open(Mean_wave_scan, 'rb') as f:
                im_b64 = base64.b64encode(f.read()).decode("utf-8")
                html_str += f'<div style="display:inline-block;max-width:100%;padding:10px"><img src="data:image/png;base64,{im_b64}" width="300" alt="Mean waves"></div>'
        else:
            html_str += '<div style="display:inline-block;width:300px"></div>'

        # Add the table data to the HTML
        if table_data is not None:
            table_html = '<div style="display:inline-block;padding:10px;">'
            table_html += '<table border="1">'
            cols = list(table_data.columns)
            max_widths = [len(str(c)) for c in cols]
            for _, row in table_data.iterrows():
                for i, val in enumerate(row):
                    max_widths[i] = max(max_widths[i], len(str(val)))
            table_html += '<tr>'
            for i, name in enumerate(cols):
                width = max_widths[i] + 10
                table_html += f'<th style="width:{width}px">{name}</th>'
            table_html += '</tr>'
            for _, row in table_data.iterrows():
                table_html += '<tr>'
                for i, val in enumerate(row):
                    width = max_widths[i] + 10  # add some padding
                    try:
                        col_name = cols[i]
                        if col_name in digitized_value_cols:
                            if val == '':
                                cell_style = 'background-color: white'
                            else:
                                cell_style = digitized_cell_background_style(
                                    row.get("Word", ""),
                                    float(row["Value"]),
                                    float(val),
                                )
                            cell_html = f'<td style="width:{width}px;{cell_style}">{val}</td>'
                        else:
                            cell_html = f'<td style="width:{width}px">{val}</td>'
                    except Exception:
                        cell_html = f'<td style="width:{width}px">{val}</td>'
                    table_html += cell_html
                table_html += '</tr>'
            table_html += '</table></div>'
            html_str += table_html

        else:
            html_str += '<div style="display:inline-block;max-width:100%"></div>'

        html_str += '<br>'

    html_str += '</div>'
    # Finish building the HTML string
    html_str += '</body></html>'

    return html_str


def generate_html_from_pkl():
    """Generates a html file from the previously processed pickle files"""

    # Loading lists from the saved file
    pickle_file = toml.load("config.toml")["pickle"]["segmented_data"]
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    if len(data) == 5:
        scan_paths, Digitized_scans, Annotated_scans, Text_data, Mean_wave_scans = data
    else:
        scan_paths, Digitized_scans, Annotated_scans, Text_data = data
        Mean_wave_scans = [None] * len(scan_paths)

    html_str = generate_html(
        scan_paths,
        Annotated_scans,
        Digitized_scans,
        Text_data,
        mean_wave_scans=Mean_wave_scans,
    )
    with open('generated_segmented_data.html', 'w') as f:
        f.write(html_str)


if __name__ == "__main__":
    generate_html_from_pkl()
