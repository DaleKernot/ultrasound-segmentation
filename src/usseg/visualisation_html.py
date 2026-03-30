import base64
import pickle
import toml


def generate_html(scans, annotated_scans, digitized_scans, tables):
    # Check if the number of scan paths and data tables match
    if len(scans) != len(tables):
        raise ValueError("The number of scan paths and data tables do not match.")

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
    for scan_path, Annotated_scan, Digitized_scan, table_data in zip(scans, annotated_scans, digitized_scans, tables):
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

        # Add the table data to the HTML
        if table_data is not None:
            table_html = '<div style="display:inline-block;padding:10px;">'
            table_html += '<table border="1">'
            max_widths = [0] * len(table_data.columns)
            for _, row in table_data.iterrows():
                for i, val in enumerate(row):
                    max_widths[i] = max(max_widths[i], len(str(val)))
                table_html += '<tr>'
                for i, val in enumerate(row):
                    width = max_widths[i] + 10  # add some padding
                    try:
                        if i == table_data.columns.get_loc('Digitized Value'):
                            if val == '':
                                cell_style = 'background-color: white'
                            else:
                                dv = float(val)
                                extracted = float(row['Value'])
                                ref = abs(extracted)
                                if ref == 0:
                                    # Avoid divide-by-zero; only treat as exact match if both are zero.
                                    diff_ratio = 0.0 if dv == 0 else float('inf')
                                else:
                                    diff_ratio = abs(dv - ref) / ref

                                if diff_ratio < 0.05:
                                    cell_style = 'background-color: green'
                                elif diff_ratio < 0.1:
                                    cell_style = 'background-color: orange'
                                else:
                                    cell_style = 'background-color: red'
                            cell_html = f'<td style="width:{width}px;{cell_style}">{val}</td>'
                        else:
                            cell_html = f'<td style="width:{width}px">{val}</td>'
                    except:
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
        scan_paths, Digitized_scans, Annotated_scans, Text_data = pickle.load(f)

    html_str = generate_html(scan_paths, Annotated_scans, Digitized_scans, Text_data)
    with open('generated_segmented_data.html', 'w') as f:
        f.write(html_str)


if __name__ == "__main__":
    generate_html_from_pkl()
