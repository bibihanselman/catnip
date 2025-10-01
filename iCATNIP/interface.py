'''
The main app file for CATNIP.
'''

# --------------------------- APP SETUP ---------------------------

from dash_extensions.enrich import (DashProxy, DashBlueprint, Output, Serverside, ServersideOutputTransform,
                                    FileSystemBackend, Input, State, Trigger, TriggerTransform, MultiplexerTransform,
                                    MATCH, ALL, ctx, dcc, html, no_update)
# Forum on serverside caching: https://community.plotly.com/t/show-and-tell-server-side-caching/42854/90
# Latest documentation: https://www.dash-extensions.com/transforms/serverside_output_transform
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash import _dash_renderer
_dash_renderer._set_react_version("18.2.0")
# Necessary for dmc to work; see https://www.dash-mantine-components.com/getting-started

# For reading/downloading files
import base64
import io
import zipfile

# Debugging stuff
import traceback
import sys

# My modules
from .image_preview import *
from .deproject import deproject
from . import profiles

# Initialize the app
external_stylesheets = [dbc.themes.LUX] # A fitting theme :)
backend = FileSystemBackend('backend')

bp = DashBlueprint(transforms=[ServersideOutputTransform(backends=[backend]), TriggerTransform(), MultiplexerTransform()])

# Global settings (might make customizable in the future 12/10/24)
GLOBALS = {
    'normalization': [0.03,100],
    'colorbar': [0,1],
    'sigma': 2,
    'smoothing': 0,
    'stretch': 'linear',
    #'parameters': [],
    'parameters': {
        'linear': [],
        'log': [],
        'sqrt': [],
        'squared': [],
        'asinh': [0.1],
        'asinhlin': [0.1,0.5,0.7]
    },
    'crop': 100,
    'color': 'Blues_r',
    'line_color': 'blue',
    'distance': 10,
    'ticks': 'pixels',
}

START_MSG = 'No images uploaded. Click the "Add image" button to add an image to the dashboard!'

# --------------------------- APP LAYOUT ---------------------------

# Prepare plot modal panels
profile_panel = [
    dbc.Row([
        dbc.Col([
            dbc.Label('Select a disk:'),
            dbc.RadioItems(
                id='profile-disk-options',
                options=[],
                value='',
                inline=True
            )
        ]),
        dbc.Col([
            dbc.Label('Select image(s):'),
            dbc.Checklist(
                id='profile-image-options',
                options=[],
                value=[],
                inline=True
            )
        ]),
    ]),
    html.Div(
        [
            html.Hr(),
            dbc.Row([
                html.H5('Plot Settings'),
                # Profile subplots
                dbc.Col([
                    dbc.Label('Select profiles to plot:'),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label('Radial:', id='rad-label'),
                            dbc.Checklist(
                                id='rad-prof-selection',
                                options=[],
                                value=[],
                                inline=True
                            )
                        ]),
                        dbc.Col([
                            dbc.Label('Azimuthal:', id='az-label'),
                            dbc.Checklist(
                                id='az-prof-selection',
                                options=[],
                                value=[],
                                inline=True
                            )
                        ])
                    ]),
                    html.Div([
                        dbc.Label('Radial Step:'), # Sampling method for radial cuts.
                        dbc.RadioItems(
                            id='rad-step',
                            options=[
                                {'label': 'Resolution', 'value': 'resolution'},
                                {'label': 'Nyquist', 'value': 'nyquist'},
                                {'label': 'Pixel', 'value': 'pixel'}
                            ],
                            value='resolution',
                            inline=True
                        ),
                    ], id='rad-step-div'),
                    html.Div([
                        dbc.Label('PA Step:'), # Increment by which PA values are sampled during azimuthally averaged radial profile construction.
                        # Make this one only appear w/ avgrad 3/14/25
                        dbc.Input(value=5, type='number', id='pa-step')
                    ], id='pa-step-div'),
                ]),
                # Other subplots
                dbc.Col([
                    dbc.Label('Additional subplots:'),
                    dbc.Checklist(
                        id='subplot-selection',
                        options=[
                            {'label': 'Azimuthally averaged radial profile(s)', 'value': 'avgrad'},
                            {'label': 'Polar projections', 'value': 'rad_az'},
                            {'label': 'Deprojected imagery', 'value': 'deprj_ims'},
                            {'label': 'On sky imagery', 'value': 'or_ims'},
                        ],
                        value=[],
                        inline=True
                    ),
                    # Image-related options (should only show if imagery plots are selected)
                    dbc.Card([
                        dbc.CardHeader('Additional options for disk imagery (if applicable):'),
                        dbc.CardBody([
                            html.Div([
                                dmc.Switch('AU Scalebar', id='scalebar-switch', checked=True),
                                html.Div([
                                    dbc.Label('AU:'), # Length of the au scalebar.
                                    dbc.Input(value=30, type='number', id='au-scale')
                                ], id='au-scalebar'),
                                html.Div([
                                    html.Hr(),
                                    dmc.Switch('Annotate profiles', id='profile-plot-annotate-switch')  # Whether to draw indicators on the imagery panels that correspond to the selected profile cuts.
                                ], id='profile-plot-annotate-div')
                            ], id='imagery-options'),
                        ])
                    ])
                ]),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col(dbc.Alert('Advanced display features coming soon...', color='primary'))
            ]),
            html.Hr(),
            dbc.Row([
                # Outputs
                dbc.Col([
                    dbc.Label('What would you like to generate?'),
                    dbc.Alert('Select a profile cut or subplot to generate a data product.', color='danger', id='plot-submit-alert')
                ]),
                dbc.Col([
                    dbc.Checklist(
                        id='output-selection',
                        options=[
                            {'label': 'Multipanel plot', 'value': 'plot'},
                            {'label': 'Tabular profile data (csv)', 'value': 'csv'}
                        ],
                        value=[],
                        inline=True
                    ),
                    dbc.Button('Go!', id='profile-plot-submit-button'), #enable if >=1 output selected
                ])
            ])
        ],
        id='profile-plot-settings') # includes submit button; available once a disk and >= 1 image have been selected
]

# App layout
bp.layout = dmc.MantineProvider(dbc.Container([
    html.Img(src='assets/newcatniplogo.png', style={'height':'25%', 'width':'25%'}),
    html.Br(),
    dbc.Row([
        dbc.Button('Add image', id='upload-button', className='me-2'),
        # Upload dialog
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle('Add image'), close_button=True, id='upload-header'),
                dbc.ModalBody([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            html.A('Select FITS file'),
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        }
                    ),
                    html.Div(id='upload-message'),
                    html.Div(
                        [
                            html.Hr(),
                            dbc.Row([
                                # Input disk name, tag
                                dbc.Col([
                                    dbc.Label('Disk:'),
                                    dbc.Input(id='disk', type='text', placeholder='e.g., "HD 100453"'),
                                    dbc.Label('Image Tag:'),
                                    dbc.Input(id='tag', type='text', placeholder='e.g., "Scattered light"')
                                ]),
                                # Input tracer
                                dbc.Col([
                                    dbc.Label('Instrument:'),
                                    dcc.Dropdown(
                                        id='instrument',
                                        options=[
                                            {'label': 'VLT-SPHERE', 'value': 'SPHERE'},
                                            {'label': 'GPI', 'value': 'GEMINI'},
                                            {'label': 'Subaru', 'value': 'SUBARU'},
                                            {'label': 'ALMA', 'value': 'ALMA'}
                                        ],
                                        value='SPHERE',
                                        clearable=False,
                                    ),
                                    # Stokes component (sl only)
                                    html.Div(id='stokes-div')
                                ])
                            ]),
                            html.Hr(),
                            html.Div(id='upload-error', style={'color': 'red'}),
                            dbc.Row([
                                dbc.Button('Go!', id='submit-button', className='me-2') #n_clicks triggers callback! (close the dialog)
                            ])
                        ], 
                        id='upload-options',
                        style={'display': 'none'}
                    ),
                ])
            ],
            id='upload-modal',
            is_open=False,
            size='lg',
            keyboard=False,
            backdrop='static'
        ),
        dbc.Button('Generate plot', id='plot-button', className='me-2'),
        # Plot dialog
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle('Generate plot'), close_button=True, id='plot-header'),
                dbc.ModalBody([
                    dbc.Alert('Choose a plot type by selecting one of the tabs below!', color='info'),
                    # I like dmc tabs for this because of the vertical orientation
                    dmc.Tabs(
                        [
                            dmc.TabsList(
                                [
                                    dmc.TabsTab('Multipanel profiles', value='cat'),
                                    dmc.TabsTab('Image gallery', value='alex'),
                                    dmc.TabsTab('More plot types coming soon...', value='bruh', disabled=True),
                                ]
                            ),
                            dmc.TabsPanel(profile_panel, value='cat'),
                            dmc.TabsPanel('Coming soon... once I get the other one working again...', value='alex'),
                        ],
                        orientation='vertical',
                        id='plot-tabs'
                    )
                ])
            ],
            id='plot-modal',
            is_open=False,
            size='xl',
            keyboard=False,
            backdrop='static'
        )
    ]),
    html.Hr(),

    html.Div(START_MSG, id='dashboard'), # Where the accordion will be placed
    dcc.Store(id='master', data={}, storage_type='local'),
    dcc.Store(id='master-diskwide', data={}, storage_type='local'), # For disk-wide info: pa, incl, distance, PAlist (rad), radlist (az)
    dcc.Store(id='last-index', storage_type='local'), # Trigger for updating layout (only want to update after upload or deleting panel).
    dcc.Download(id='download'), # Where files will be downloaded!
    
    # Clear persistence button (testing)
    dbc.Button('Clear Persistence', id='persistence-clear')
], fluid=True))

# Clearing persistent values (for testing)
@bp.callback(
    Output('master', 'data'),
    Output('master-diskwide', 'data'),
    Output('last-index', 'data'),
    Input('persistence-clear', 'n_clicks'),
    prevent_initial_call=True
)
def clear_persistence(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return {}, {}, []

# --------------------------- SUBMIT FILE ---------------------------

# Update the status of the submit button
@bp.callback(
    Output('submit-button', 'disabled'),
    Input('upload-data', 'filename'),
    Input('disk', 'value'),
    Input('tag', 'value')
)
def update_submit_button(filename, disk, tag):
    # Requirements: file must exist and have .fit or .fits extension;
    # Disk and tag fields must be non-empty
    if not filename:
        return True
    return not (filename.endswith(('.fit', '.fits')) and disk and tag)

# Update the upload status message
@bp.callback(
    Output('upload-message', 'children'),
    Output('upload-message', 'style'),
    Output('upload-options', 'style'),
    Input('upload-data', 'filename'),
)
def update_upload_message(filename):
    if not filename:
        return 'No file uploaded.', {'color': 'black'}, {'display': 'none'}
    elif not filename.endswith(('.fit', '.fits')):
        return 'Invalid file extension. Please upload a .fit or .fits file.', {'color': 'red'}, {'display': 'none'}
    return f'File uploaded: {filename}', {'color': 'black'}, {'display': 'block'}

# Toggle the upload dialog
@bp.callback(
    Output('upload-modal', 'is_open'),
    Trigger('upload-button', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_upload():
    return True

# Toggle stokes options for scattered light
@bp.callback(
    Output('stokes-div', 'children'),
    Input('instrument', 'value')
)
def toggle_stokes(instr):
    sl = [
        dbc.Label('Stokes Component:'),
        dbc.RadioItems(
            id='tracer',
            options=[
                {'label': 'I', 'value': 'I'},
                {'label': 'Q_phi', 'value': 'Q_phi'},
                {'label': 'U_phi', 'value': 'U_phi'},
                {'label': 'Q', 'value': 'Q'},
                {'label': 'U', 'value': 'U'},
                {'label': 'LP_I', 'value': 'LP_I'}
            ],
            value='Q_phi',
            inline=True
        )
    ]
    mm = [
        dbc.Label('Tracer:'),
        dbc.RadioItems(
            id='tracer',
            options=[
                {'label': 'Continuum', 'value': 'continuum'},
                {'label': 'Line', 'value': 'line', 'disabled': True},
            ],
            value='continuum',
            inline=True
        ),
        dbc.Alert('ALMA line emission data will be supported in a future release.', color='info'),
    ]
    return mm if instr == 'ALMA' else sl

# When an uploaded file is submitted, read the file and add the image to the layout
@bp.callback(
    Output('master', 'data'),
    Output('master-diskwide', 'data'),
    Output('last-index', 'data'),
    Output('upload-modal', 'is_open'),
    Output('upload-error', 'children'),
    Output('upload-data', 'filename'),
    Input('submit-button', 'n_clicks'),
    State('upload-data', 'contents'),
    State('disk', 'value'),
    State('tag', 'value'),
    State('tracer', 'value'),
    State('instrument', 'value'),
    State('master', 'data'),
    State('master-diskwide', 'data'),
    prevent_initial_call=True,
    running=[
        (Output('submit-button', 'disabled'), True, False),
        (Output('upload-data', 'disabled'), True, False),
        (Output('upload-header', 'close_button'), False, True),
        (Output('upload-error', 'children'), '', ''),
        (Output('submit-button', 'children'),
         [dbc.Spinner(size='sm'), ' Go!'],
         'Go!')
    ]
)
def submit_file(n, contents, name, tag, tracer, instr, master, master_diskwide):
    try:
        # Read the data and header
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        data, header = fits.getdata(io.BytesIO(decoded), header=True)

        # If cont, choose the correct slice bc ALMA files are 4d
        if tracer == 'continuum':
            data = data[0,0,:,:]
        # If scattered light, extract the correct slice from the stokes cube
        else:
            data = data[header['STOKES'].split(',').index(tracer)]

        # Convert data to 64-bit for image rotation
        data = skimage.util.img_as_float64(data)
    except:
        # Get the exception information
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        # Print the stack trace
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        
        return no_update, no_update, no_update, no_update, 'Error reading file. Did you select the correct tracer?', no_update

    # Important image information (thanks Cat)
    try:
        if instr == 'ALMA':
            date = header['DATE-OBS'].split('T')[0]
            pixscale = None
            for row in header['HISTORY']:
                if 'cell' in row:
                    #https://casaguides.nrao.edu/index.php?title=Image_Continuum#Image_Parameters
                    pixscale = (float(row.split("cell")[1].split("=")[1].split("arcsec")[0].replace("'","").replace("[","").replace("]","")) * u.arcsec).to(u.si.mas).value
                    break
            if pixscale is None:
                pixscale = (abs(float(header['CDELT1'])) * u.deg).to(u.si.mas).value
            xcen, ycen = (header['CRPIX1'], header['CRPIX2'])
            resolution = (header['BMAJ'] + header['BMIN'])/2 * u.deg
            px_bin = int(np.floor(resolution.to(u.arcsec)/((pixscale * u.si.mas).to(u.arcsec))).value)
            
            # Get the frequency band
            # https://www.eso.org/public/teles-instr/alma/receiver-bands/
            bands = {'Band 1': [35, 50],
                     'Band 2': [67, 84],
                     'Band 3': [84, 116],
                     'Band 4': [125, 163],
                     'Band 5': [163, 211],
                     'Band 6': [211, 275],
                     'Band 7': [275, 373],
                     'Band 8': [385, 500],
                     'Band 9': [602, 720],
                     'Band 10': [787, 950]}
            for b in bands:
                if bands[b][0] <= (header['RESTFRQ'] * 1e-9) <= bands[b][1]: band = b

            beam = {
                'BMAJ': header['BMAJ']*3600,
                'BMIN': header['BMIN']*3600,
                'BPA': header['BPA']*3600
            }
                                
        elif instr == 'SPHERE':
            date = header['DATE']
            pixscale = header['SCALE']
            xcen, ycen = (header['STAR_X'], header['STAR_Y'])
            #https://courses.lumenlearning.com/suny-physics/chapter/27-6-limits-of-resolution-the-rayleigh-criterion/
            rayleigh_crit = 1.22 * (header['WAVELE'] * u.si.micron).to(u.meter)/(8.2 * u.meter) * u.si.rad
            px_bin = int(np.floor(rayleigh_crit.to(u.si.mas)/(pixscale * u.si.mas)))
            band = header['FILTER']
    except:
        return no_update, no_update, no_update, no_update, 'Error stripping FITS header. Did you select the correct instrument?', no_update

    # Store all image information in a dict
    imdat = {
        'Date': date,
        'Name': name,
        'Source': instr,
        'Tag': tag,
        'Tracer': tracer if instr == 'ALMA' else 'scattered light',
        'Stokes Component': tracer if instr != 'ALMA' else None,
        'Pixel Scale': pixscale,
        'Center': [int(ycen), int(xcen)],
        'Dimensions': [data.shape[0], data.shape[1]],
        ### Remaining strip_header attributes not used in the callbacks, but needed for profile generation code ###
        'Pixel Bin': px_bin,
        'Band': band,
        ###########################################################################################################
        'Index': f'{name}-{tag}',
        'Data': data,
        'Normalization': GLOBALS['normalization'],
        'Colorbar': GLOBALS['colorbar'],
        'Sigma': GLOBALS['sigma'],
        'Smoothing': GLOBALS['smoothing'],
        'Stretch': GLOBALS['stretch'],
        'Parameters List': GLOBALS['parameters'],
        'Parameters': GLOBALS['parameters'][GLOBALS['stretch']],
        'Crop': min(int(data.shape[0]),300), # Set an upper bound for initial cropping to lower load time. Might make this bound a global setting. 3/26/25
        'Color': GLOBALS['color'],
        'Line Color': GLOBALS['line_color'],
        'Distance': GLOBALS['distance'],
        'Ticks': GLOBALS['ticks']
    }
    # Beam (radio only)
    if instr == 'ALMA':
        imdat['Beam'] = beam

    # Update the master dict
    if name not in master:
        master[name] = {}
        master_diskwide[name] = {
            'Name': name,
            'PA': None,
            'Inclination': None,
            'Distance': GLOBALS['distance'], #Simbad query? 12/11/24
            'PA List': [],
            'Radius List': [],
            'Unit': 'mas'
        }
    master[name][tag] = imdat

    # Save the index to open
    index = [name, tag]
    
    # Trigger layout update (not elegant, but it works...) and close the dialog
    return Serverside(master), Serverside(master_diskwide), Serverside(index), False, '', ''

# Create accordion entry template
def create_display_template(imdat):
    disk, tag = imdat['Name'], imdat['Tag']
    return dbc.AccordionItem([
        dbc.Row([
            dbc.Col(
                [
                    # The graph
                    dcc.Loading(
                        type='default',
                        overlay_style={"visibility":"visible", "filter": "blur(2px)"},
                        children=[dcc.Graph(id={'type': 'on-sky', 'index': imdat['Index']}, figure=update_image(imdat['Data'], imdat, title='On Sky'), config={'staticPlot': True})]
                    ),
                    # Toggle crosshair
                    dmc.Switch(id={'type': 'crosshair-switch', 'index': imdat['Index']},
                             label='Show center crosshair',
                             checked=False)
                ],
                width=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                #normalization
                dbc.Label('Normalization Bounds:'),
                dcc.RangeSlider(0.01, 100, value = imdat['Normalization'], id = {'type': 'normalization', 'index': imdat['Index']},
                              tooltip = {'placement': 'bottom', 'always_visible': True},
                              updatemode='drag'),
                #colorbar bounds
                dbc.Label('Colorbar Bounds:'),
                dcc.RangeSlider(0, 1, value = imdat['Colorbar'], id = {'type': 'colorbar', 'index': imdat['Index']},
                              tooltip = {'placement': 'bottom', 'always_visible': True},
                              updatemode='drag'),
                #sigma clipping
                dbc.Label('Background Sigma:'),
                dcc.Slider(0, 5, value = imdat['Sigma'], id = {'type': 'sigma', 'index': imdat['Index']},
                          tooltip = {'placement': 'bottom', 'always_visible': True},
                          updatemode='drag'),
                #smoothing
                dbc.Label('Smoothing Parameter:'),
                dcc.Slider(0, 5, value = imdat['Smoothing'], id = {'type': 'smoothing', 'index': imdat['Index']},
                          tooltip = {'placement': 'bottom', 'always_visible': True},
                          updatemode='drag'),
                #colorbar scaling
                dbc.Label('Stretch:'),
                dcc.Dropdown(
                    id={'type': 'stretch', 'index': imdat['Index']},
                    options=[
                        {'label': 'Linear', 'value': 'linear'},
                        {'label': 'Log', 'value': 'log'},
                        {'label': 'Sqrt', 'value': 'sqrt'},
                        {'label': 'Squared', 'value': 'squared'},
                        {'label': 'Asinh', 'value': 'asinh'},
                        {'label': 'Asinh Lin', 'value': 'asinhlin'}
                    ],
                    value=imdat['Stretch'],
                    clearable=False
                ),
                #scaling parameters (for certain stretches)
                html.Div(generate_stretch_sliders(imdat['Parameters'], imdat['Index'], imdat['Stretch']), id={'type': 'stretch-params', 'index': imdat['Index']})
            ])), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                #cropping
                dbc.Label('Crop (px):'),
                dbc.Input(id={'type': 'crop', 'index': imdat['Index']}, type='number',
                          value=imdat['Crop'], debounce=True),
                #color
                dbc.Label('Colormap:'),
                dcc.Dropdown(
                    id={'type': 'color', 'index': imdat['Index']},
                    options={
                        'Blues_r': 'Blue',
                        'Purples_r': 'Purple',
                        'Greens_r': 'Green',
                        'Oranges_r': 'Orange',
                        'Reds_r': 'Red',
                        'Grays_r': 'Gray',
                        'inferno': 'Inferno',
                        'magma': 'Magma',
                        'plasma': 'Plasma',
                        'viridis': 'Viridis'
                    },
                    value=imdat['Color'],
                    clearable=False
                ),
                #line color
                dbc.Label(html.Span('Line Color:',
                                    style={"textDecoration": "underline", "cursor": "pointer"},
                                    id={'type': 'line-color-label', 'index': imdat['Index']})),
                dbc.Tooltip(
                    'For the "multipanel profiles" plotting option, the line color of profiles for this image.',
                    target={'type': 'line-color-label', 'index': imdat['Index']}
                ),
                dcc.Dropdown(
                    id={'type': 'line-color', 'index': imdat['Index']},
                    options={
                        'blue': 'Blue',
                        'purple': 'Purple',
                        'green': 'Green',
                        'orange': 'Orange',
                        'red': 'Red',
                        'gray': 'Gray',
                    },
                    value=imdat['Line Color'],
                    clearable=False
                ),
                #center coords
                dbc.Label('Center Coordinates (px):'),
                dbc.Row(
                    [
                      dbc.Col(dbc.Input(id={'type': 'centerx', 'index': imdat['Index']}, type='number', value=imdat['Center'][1], debounce=True), width=6),
                      dbc.Col(dbc.Input(id={'type': 'centery', 'index': imdat['Index']}, type='number', value=imdat['Center'][0], debounce=True), width=6)
                    ],
                    id={'type': 'center', 'index': imdat['Index']}
                ),
                #ticks
                dbc.Label('Axes:'),
                dcc.RadioItems(
                    id={'type': 'ticks', 'index': imdat['Index']},
                    options=[
                      {'label': 'Pixels', 'value': 'pixels'},
                      {'label': 'Arcseconds', 'value': 'angles'},
                      {'label': 'au', 'value': 'spatial'},
                    ],
                    value=imdat['Ticks']
                ),
                html.Div(f'Distance: {imdat["Distance"]} pc', id={'type': 'distance-div', 'index': imdat['Index']}),
                dbc.Alert(f'{disk}\'s distance may be modified in the "Profiles" tab for this disk entry.', color='info')
            ])), width=3)
        ]),
        dbc.Row([
            dbc.Col(
                dcc.ConfirmDialogProvider(
                    children=dbc.Button('Remove image', color='danger'),
                    id={'type': 'delete', 'index': imdat['Index']},
                    message=f'Are you sure you want to remove this image ({disk} - {tag}) from the layout? All changes will be lost.',
                    submit_n_clicks=0
                )
            )
        ]),
        ],
        title=tag,
        item_id=tag
    )

def ellipse_modal(imdat):
    imdat['Color'] = 'gray'
    fig = update_image(imdat['Data'], imdat, title='On Sky')
    fig.update_layout(dragmode='drawopenpath')
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle('Fit ellipse'), close_button=True, id={'type': 'ellipse-header', 'index': imdat['Index']}),
            dbc.ModalBody(
                [
                    dbc.Alert('Draw a path to which the ellipse should be fitted.', color='info'),
                    dcc.Graph(id={'type': 'ellipse-fig', 'index': imdat['Index']},
                              figure=fig, 
                              config={'modeBarButtonsToAdd': ['drawopenpath', 'eraseshape']})
                ]
            ),
            dbc.ModalFooter(
                dbc.Button('Fit!', id={'type': 'ellipse-confirm-btn', 'index': imdat['Index']}, n_clicks=0)
            )
        ],
        id={'type': 'ellipse-modal', 'index': imdat['Index']},
        is_open=False,
        size='lg',
        keyboard=False,
        backdrop='static'
    )

def create_profile_template(imdat):
    index = imdat['Name']
    return dbc.Tab(
        [
            dbc.Row([
                dbc.Col([
                    dbc.Row(
                        # The graph
                        dcc.Loading(
                            type='default',
                            overlay_style={"visibility":"visible", "filter": "blur(2px)"},
                            id={'type': 'deproj-div', 'index': index}
                            )
                    ),
                    dbc.Row([
                        dbc.Col(
                            [
                                dbc.Label('PA (deg):'),
                                dbc.Input(id={'type': 'pa', 'index': index}, type='number', min=0, max=360, value=imdat['PA'], debounce=True)
                            ],
                            width=6),
                        dbc.Col(
                            [
                                dbc.Label('Inclination (deg):'),
                                dbc.Input(id={'type': 'incl', 'index': index}, type='number', min=0, max=90, value=imdat['Inclination'], debounce=True)
                            ],
                            width=6),
                    ]),
                    # ellipse_modal(imdat), # Ellipse fitting dialog
                    dbc.Button('Fit ellipse', id={'type': 'ellipse-modal-btn', 'index': index}, className='me-2'),
                    html.Br(),
                    dbc.Label('Distance (pc):'),
                    dbc.Input(id={'type': 'distance', 'index': index}, type='number', min=0, max=360, value=imdat['Distance'], debounce=True)
                ]),
                dbc.Col([
                    dbc.Tabs(
                        [
                            dbc.Tab(
                                [
                                    dbc.InputGroup(
                                        [
                                            dbc.Input(id={'type': 'profile-input', 'index': f'{index}:rad'}, type='number', placeholder='Insert PA...'),
                                            dbc.Button("Add", id={'type': 'profile-add', 'index': f'{index}:rad'}, n_clicks=0)
                                        ]
                                    ),
                                    dbc.ListGroup(id={'type': 'profile-list', 'index': f'{index}:rad'}, children=[]),
                                    dcc.Store(id={'type': 'profile-store', 'index': f'{index}:rad'}, data=imdat['PA List'])
                                ],
                                label='Radial',
                                tab_id='rad',
                                tab_class_name='flex-grow-1 text-center',
                                active_tab_class_name='fw-bold'
                            ),
                            dbc.Tab(
                                [
                                    dbc.InputGroup(
                                        [
                                            dbc.Input(id={'type': 'profile-input', 'index': f'{index}:az'}, type='number', placeholder='Insert radius...'),
                                            dbc.Button('Add', id={'type': 'profile-add', 'index': f'{index}:az'}, n_clicks=0)
                                        ]
                                    ),
                                    dbc.ListGroup(id={'type': 'profile-list', 'index': f'{index}:az'}, children=[]),
                                    html.Br(),
                                    dbc.Label('Units:'),
                                    dcc.Dropdown(
                                        id={'type': 'az-units', 'index': f'{index}:az'},
                                        options={
                                            'mas': 'mas',
                                            'au': 'au'
                                        },
                                        value='mas',
                                        clearable=False
                                    ),
                                    dcc.Store(id={'type': 'profile-store', 'index': f'{index}:az'}, data=imdat['Radius List'])
                                ],
                                label='Azimuthal',
                                tab_id='az',
                                tab_class_name='flex-grow-1 text-center',
                                active_tab_class_name='fw-bold'
                            )                   
                        ],
                        id={'type': 'profile-tab', 'index': index}
                    ),
                ])
            ]),
        ],
        label='Profiles',
        tab_id='profile',
        tab_class_name='flex-grow-1 text-center',
        active_tab_class_name='fw-bold'
    )

# Clear all button
clear_all_btn = dcc.ConfirmDialogProvider(
                    children=dbc.Button('Clear all', color='danger'),
                    id='clear',
                    message='You sure buddy?',
                    submit_n_clicks=0)

# Update the dashboard (!!)
@bp.callback(
    Output('dashboard', 'children'),
    Input('last-index', 'data'),
    State('master', 'data'),
    State('master-diskwide', 'data'),
    prevent_initial_call=True,
    running=[
        (Output('dashboard', 'children'), [dbc.Spinner(size='sm'), ' Please wait...'], '')
        # In lieu of a loading wrapper, which would display any time a dashboard component changes
    ]
)
def update_layout(index, master, master_diskwide):
    if not master:
        return START_MSG
    new_disk, new_tag = index if index is not None else ['','']
    return [
        clear_all_btn,
        html.Hr(),
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        dbc.Tabs(
                            [
                                dbc.Tab(
                                    [
                                        dbc.Accordion(
                                            [
                                                create_display_template(master[disk][tag])
                                                for tag in master[disk]
                                            ],
                                            active_item=(new_tag if (new_tag in master[disk] and disk == new_disk) else None)
                                        )
                                    ],
                                    label='Image Rendering',
                                    tab_id='display',
                                    tab_class_name='flex-grow-1 text-center',
                                    active_tab_class_name='fw-bold'
                                    ),
                                create_profile_template(master_diskwide[disk])
                            ],
                            active_tab='display')
                    ],
                    title=disk,
                    item_id=disk
                )
                for disk in master
            ],
            active_item=new_disk
        )
    ]

# --------------------------- DASHBOARD INTERACTIVITY ---------------------------

# Update the image preview when a parameter changes
# Future change - make axes updates a separate callback to improve performance for changes in distance or axes. 1/1/25
@bp.callback(
    Output({'type': 'on-sky', 'index': ALL}, 'figure'),
    Output({'type': 'distance-div', 'index': ALL}, 'children'),
    Output('master', 'data'),
    Input({'type': 'normalization', 'index': ALL}, 'value'),
    Input({'type': 'sigma', 'index': ALL}, 'value'),
    Input({'type': 'colorbar', 'index': ALL}, 'value'),
    Input({'type': 'crop', 'index': ALL}, 'value'),
    Input({'type': 'smoothing', 'index': ALL}, 'value'),
    State({'type': 'stretch', 'index': ALL}, 'value'),
    Input({'type': 'stretch-param', 'index': ALL}, 'value'),
    Input({'type': 'color', 'index': ALL}, 'value'),
    Input({'type': 'line-color', 'index': ALL}, 'value'),
    Input({'type': 'centerx', 'index': ALL}, 'value'),
    Input({'type': 'centery', 'index': ALL}, 'value'),
    Input({'type': 'distance', 'index': ALL}, 'value'),
    Input({'type': 'ticks', 'index': ALL}, 'value'),
    State({'type': 'on-sky', 'index': ALL}, 'figure'),
    State({'type': 'distance-div', 'index': ALL}, 'children'),
    State({'type': 'on-sky', 'index': ALL}, 'id'),
    State({'type': 'stretch-param', 'index': ALL}, 'id'),
    State({'type': 'distance', 'index': ALL}, 'id'),
    State('master', 'data'),
    prevent_initial_call=True
)
def image_preview(normalization, sigma, colorbar, crop, smoothing, stretch, params, color, line_color, xcen, ycen, distance, ticks, figs, divs, fig_ids, param_ids, dist_ids, master):
    # Get correct index and imdat
    trigger = ctx.triggered_id
    index = trigger.index
    if trigger.type == 'distance':
        disk = index
        tags = master[disk].keys()
    elif trigger.type == 'stretch-param':
        disk, tag, _ = index.split('-')
        tags = [tag]
    else:
        disk, tag = index.split('-')
        tags = [tag]
    # Make distance a separate callback so you don't have this tags nonsense 3/30/25

    # Get correct position for the fig and params
    indices = [fig_ids.index({'type': 'on-sky', 'index': f'{disk}-{tag}'}) for tag in tags]

    # Get correct position for the (disk-wide) distance (special case)
    dist_index = dist_ids.index({'type': 'distance', 'index': disk})

    # The specialer case of stretch parameters
    param_indices = {tag : [param_ids.index(id) for id in param_ids if f'{disk}-{tag}' in id['index']] for tag in tags}
    selected_params = {tag : [params[param_index] for param_index in param_indices[tag] if params[param_index] is not None] for tag in tags}

    # Plot those suckers
    for i, tag in zip(indices, tags):
        # Store parameter values in a dict a la FIGG
        new_info = {
            'Normalization': normalization[i],
            'Crop': crop[i],
            'Smoothing': smoothing[i],
            'Sigma': sigma[i],
            'Colorbar': colorbar[i],
            'Color': color[i],
            'Line Color': line_color[i],
            'Stretch': stretch[i],
            'Parameters': selected_params[tag],
            'Center': [ycen[i],xcen[i]],
            'Distance': distance[dist_index],
            'Ticks': ticks[i]
        }
        # Update the master w/ new values
        for key, val in new_info.items(): master[disk][tag][key] = val
        master[disk][tag]['Parameters List'][stretch[i]] = selected_params[tag]
    
        # Update the correct fig
        imdat = master[disk][tag]
        figs[i] = update_image(imdat['Data'], imdat, title='On Sky')
    
        # Update the distance text (may as well do it in this callback)
        # Shouldn't affect performance too much but might want to optimize. 1/1/25
        divs[i] = f'Distance: {imdat["Distance"]} pc'

    # Refresh the display
    return figs, divs, Serverside(master)

# Generate stretch parameter sliders based on stretch selection
def generate_stretch_sliders(info, index, stretch):
    # Slider template
    def slider(i, val):
        return dbc.Col([
            dbc.Label(f'Stretch Parameter {i + 1}:'),
            dcc.Slider(0, 1, value = val, id = {'type': 'stretch-param', 'index': f'{index}-{i}'},
                      tooltip = {'placement': 'bottom', 'always_visible': True},
                      updatemode='drag')
        ])

    dummy = html.Div(
        dbc.Input(
            value=None,
            id={'type': 'stretch-param', 'index': f'{index}-0'}
            ),
        style={'display': 'none'}
    )
    # The dummy is so that when there are no params for the stretch,
    # the image_preview callback will still have a triggered_id of type 'stretch-param'.
    # Likely not an optimal solution, but it works...
    return dbc.Row([slider(i, val) for i, val in enumerate(info)]) if info else [f'No parameters for {stretch} stretch.', dummy]
    # Mini issue: asinhlin doesn't accept c <= b, so find a way to generically adjust the bounds for stretch param 3

# Update list of stretch parameters
@bp.callback(
    Output({'type': 'stretch-params', 'index': ALL}, 'children'),
    Input({'type': 'stretch', 'index': ALL}, 'value'),
    State({'type': 'stretch-params', 'index': ALL}, 'children'),
    State({'type': 'stretch', 'index': ALL}, 'id'),
    State('master', 'data'),
    prevent_initial_call=True
)
def update_stretch_params(stretches, stretch_params, stretch_ids, master):
    # Get the correct index
    disk, tag = ctx.triggered_id.index.split('-')
    stretch_index = stretch_ids.index({'type': 'stretch', 'index': ctx.triggered_id.index})
    stretch = stretches[stretch_index]
    stretch_params[stretch_index] = generate_stretch_sliders(master[disk][tag]['Parameters List'][stretch], f'{disk}-{tag}', stretch)
    return stretch_params

# Toggle crosshair for image preview
@bp.callback(
    Output({'type': 'on-sky', 'index': MATCH}, 'figure'),
    Input({'type': 'crosshair-switch', 'index': MATCH}, 'checked'),
    Input({'type': 'on-sky', 'index': MATCH}, 'figure'),
    prevent_initial_call=True
)
def toggle_crosshair(checked, fig):
    fig = go.Figure(fig)
    if checked:
        shapes = [
            dict(type='line',
                xref='x domain', yref='y domain',
                x0=0.5, y0=0.48, x1=0.5, y1=0.52,
                line_color='red'),
            dict(type='line',
                xref='x domain', yref='y domain',
                x0=0.48, y0=0.5, x1=0.52, y1=0.5,
                line_color='red')
         ] # Make crosshair size customizable? 3/26/25
    else: shapes = []
    fig.layout.shapes = shapes
    fig.update_layout()
    fig = fig.to_dict()
    
    return fig

# Deprojected image preview when PA and incl are given
@bp.callback(
    Output({'type': 'deproj-div', 'index': ALL}, 'children'),
    Output('master-diskwide', 'data'),
    Input({'type': 'pa', 'index': ALL}, 'value'),
    Input({'type': 'incl', 'index': ALL}, 'value'),
    Input({'type': 'distance', 'index': ALL}, 'value'),
    Trigger({'type': 'on-sky', 'index': ALL}, 'figure'),
    State('master', 'data'), # propagate display panel changes to deprojected ims
    # Not ideal due to the delay b/w callbacks - might combine w/ image_preview into one callback. 12/31/24
    State('master-diskwide', 'data'),
    State({'type': 'deproj-div', 'index': ALL}, 'children'),
    State({'type': 'deproj-div', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def deproject_preview(pa, incl, dist, master, master_diskwide, divs, div_ids):        
    # Get correct index (disk)
    trigger = ctx.triggered_id
    if trigger.type == 'on-sky':
        disk = trigger.index.split('-')[0]
    else: disk = ctx.triggered_id.index

    # Get correct position
    i = div_ids.index({'type': 'deproj-div', 'index': disk})
    
    # Don't deproject if pa and incl aren't given
    if pa[i] is None or incl[i] is None:
        divs[i] = dbc.Alert('To preview the deprojected image(s), specify the disk\'s PA and inclination.', color='info')
        return divs, no_update
    
    annotations = dmc.Switch(id={'type': 'annotations-switch', 'index': disk},
                             label='Show annotations',
                             checked=False)
    comps = [annotations]

    for imdat in master[disk].values():
        tag = imdat['Tag']
        
        # Deproject image data
        deproj = deproject(imdat['Data'], imdat, pa[i], incl[i])
    
        # New center for displaying the deprojected image
        imdat['Center'][0] = int(imdat['Center'][0] / math.cos(math.radians(incl[i])))
        
        # Redraw the figure w/ processing (like on sky)
        graph = dcc.Graph(id={'type': 'deproj', 'index': f'{disk}-{tag}'},
                          figure=update_image(deproj, imdat, title=tag),
                          config={'staticPlot': True})
        comps.append(graph)

    # Update the correct div
    divs[i] = comps

    # Update the disk-wide master
    new_info = {
        'PA': pa[i],
        'Inclination': incl[i],
        'Distance': dist[i]
    }
    for key, val in new_info.items(): master_diskwide[disk][key] = val
    
    return divs, Serverside(master_diskwide)

# Add profile to store container (rad or az)
@bp.callback(
    Output({'type': 'profile-input', 'index': MATCH}, 'value'),
    Output({'type': 'profile-store', 'index': MATCH}, 'data'),
    Trigger({'type': 'profile-add', 'index': MATCH}, 'n_clicks'),
    State({'type': 'profile-input', 'index': MATCH}, 'value'),
    State({'type': 'profile-store', 'index': MATCH}, 'data'),
    prevent_initial_call=True
)
def add_prof(value, data):    
    if not value:
        raise PreventUpdate
    if value not in data:
        data.append(value)
    value = None
    data.sort()

    return value, Serverside(data)

# Render the listgroups for rad or az profiles based on the contents of the respective containers
@bp.callback(
    Output({'type': 'profile-list', 'index': ALL}, 'children'),
    Output('master-diskwide', 'data'),
    Input({'type': 'profile-store', 'index': ALL}, 'data'),
    State('master-diskwide', 'data'),
    State({'type': 'profile-list', 'index': ALL}, 'children'),
    State({'type': 'profile-store', 'index': ALL}, 'id'),
    State({'type': 'az-units', 'index': ALL}, 'value'),
    State({'type': 'az-units', 'index': ALL}, 'id'),
    prevent_inital_call=True
)
def update_prof_lists(data, master_diskwide, lists, data_ids, units, unit_ids):
    # Triggers are funny here (all of the stores will trigger) so doing a for loop.
    # This may present performance issues for large samples. 1/13/25

    # Add a list group item to display the value and a remove button
    def item(n, val):
        return dbc.ListGroupItem(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Div(val)),
                        dbc.Col(dbc.Button('x',id={'type': 'profile-delete', 'index': f'{index}-{n}'}, color='danger', n_clicks=0))
                    ],
                    justify='between'
                )
            ],
        )
    
    for dat, id in zip(data, data_ids):
        index = id['index']
        i = data_ids.index({'type': 'profile-store', 'index': index})
    
        # Update the disk-wide master
        disk, prof_type = index.split(':')
        if prof_type == 'rad':
            master_diskwide[disk]['PA List'] = dat
        else:
            master_diskwide[disk]['Radius List'] = dat
            # Update the unit
            unit_index = unit_ids.index({'type': 'az-units', 'index': index})
            master_diskwide[disk]['Unit'] = units[unit_index]

        lists[i] = [item(n, val) for n, val in enumerate(dat)]
    
    return lists, Serverside(master_diskwide)

# Remove profile from store container (rad or az)
@bp.callback(
    Output({'type': 'profile-store', 'index': ALL}, 'data'),
    Input({'type': 'profile-delete', 'index': ALL}, 'n_clicks'),
    State({'type': 'profile-store', 'index': ALL}, 'data'),
    State({'type': 'profile-store', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def remove_prof(n, data, ids):
    index = ctx.triggered_id.index
    store_index, i = index.split('-')
    store_index = ids.index({'type': 'profile-store', 'index': store_index}) #super cheesy but it works
    i = int(i)

    trigger = ctx.triggered[0]['value']
    if trigger == 0:
        raise PreventUpdate
        
    del data[store_index][i]

    return [Serverside(dat) for dat in data]

# Update the deprojected image annotations (rad/az)
@bp.callback(
    Output({'type': 'deproj', 'index': ALL}, 'figure'), #{disk}-{tag}
    Input({'type': 'profile-tab', 'index': ALL}, 'active_tab'), #disk
    Input({'type': 'annotations-switch', 'index': ALL}, 'checked'), #disk
    Input({'type': 'profile-list', 'index': ALL}, 'children'), #{disk}:{type}
    Trigger({'type': 'deproj-div', 'index': ALL}, 'children'), #disk
    State({'type': 'az-units', 'index': ALL}, 'value'), #{disk}:az
    State({'type': 'annotations-switch', 'index': ALL}, 'id'),
    State({'type': 'profile-store', 'index': ALL}, 'data'),
    State({'type': 'profile-store', 'index': ALL}, 'id'),
    State({'type': 'deproj', 'index': ALL}, 'figure'),
    State({'type': 'deproj', 'index': ALL}, 'id'),
    State('master', 'data'),
    prevent_initial_call=True
)
def annotate(tabs, switches, lists, units, switch_ids, data, data_ids, figs, fig_ids, master):
    # Get the right figs to change
    trigger = ctx.triggered_id
    if trigger.type == 'profile-list':
        disk = trigger.index.split(':')[0]
    else: disk = trigger.index
    tags = master[disk].keys()
    fig_indices = [fig_ids.index({'type': 'deproj', 'index': f'{disk}-{tag}'}) for tag in tags]
    imdats = [master[disk][tag] for tag in tags] # muy importante for later

    # Prepare annotations or clearing depending on switch state
    disk_index = switch_ids.index({'type': 'annotations-switch', 'index': disk}) # Should match up w/ tab indices
    switch = switches[disk_index]
    if switch:
        # Get correct profile-store for drawing the annotations
        tab = tabs[disk_index]
        store = data[data_ids.index({'type': 'profile-store', 'index': f'{disk}:{tab}'})]

        # Prepare the shapes
        shapes_list = []
        for imdat in imdats:
            crop = int(imdat['Crop']/2)
            if tab == 'rad':
                shapes = [
                    dict(type='line',
                        xref='x', yref='y',
                        x0=crop, y0=crop, x1=crop+int(crop*math.cos(math.radians(180 - val))), y1=crop+int(crop*math.sin(math.radians(180 - val))),
                        line_color='red') # might make color customizable 12/29/24
                    for val in store]
            elif tab == 'az':
                # Convert the az prof entries to pixels
                unit = units[disk_index]
                if unit == 'mas':
                    store = [int(val / imdat['Pixel Scale']) for val in store]
                elif unit == 'au':
                    pix_dist = imdat['Pixel Scale'] / 1000 * imdat['Distance'] # arcsec / pixel * parsec ~= au / pixel
                    store = [int(val / pix_dist) for val in store]
                
                shapes = [
                    dict(type='circle',
                        xref='x', yref='y',
                        x0=crop-val, y0=crop-val, x1=crop+val, y1=crop+val,
                        line_color='red') # might make color customizable 12/29/24
                    for val in store]
                
            shapes_list.append(shapes)
    else: shapes_list = [[] for tag in tags]
        
    # Draw!
    for i, shapes in zip(fig_indices, shapes_list):
        fig = go.Figure(figs[i])
        # Thanks Google. For some reason fig.update_layout(shapes=[]) doesn't clear the shapes, so doing this instead!
        fig.layout.shapes = shapes
        fig.update_layout()
        figs[i] = fig.to_dict()
    
    return figs

# Unit conversion for az profiles
@bp.callback(
    Output({'type': 'profile-store', 'index': MATCH}, 'data'),
    Input({'type': 'az-units', 'index': MATCH}, 'value'),
    State({'type': 'profile-store', 'index': MATCH}, 'data'),
    State('master-diskwide', 'data'),
    prevent_initial_call=True
)
def unit_convert(unit, data, master_diskwide):
    disk = ctx.triggered_id.index.split(':')[0]
    dist = master_diskwide[disk]['Distance']
    # mas / au = mas / ('' * pc) = (mas / '') / pc = 1000 / pc
    factor = 1000 / dist
    
    if unit == 'mas' and master_diskwide[disk]['Unit'] == 'az':
        data *= factor
    elif unit == 'az' and master_diskwide[disk]['Unit'] == 'mas':
        data /= factor

    return data

# Remove an image from the layout
@bp.callback(
    Output('master', 'data'),
    Output('master-diskwide', 'data'),
    Output('last-index', 'data'),
    Trigger({'type': 'delete', 'index': ALL}, 'submit_n_clicks'),
    State('master', 'data'),
    State('master-diskwide', 'data'),
    prevent_initial_call=True
)
def remove_image(master, master_diskwide):
    # Get the button index that triggered the callback
    trigger = ctx.triggered[0]['value']
    if trigger == 0:
        raise PreventUpdate # if the button hasn't been clicked
    
    # Remove the image
    disk, tag = ctx.triggered_id.index.split('-')
    del master[disk][tag]
    # If there are no more images for the disk, remove the disk from both master dicts
    if not master[disk]:
        del master[disk]
        del master_diskwide[disk]
    
    # Refresh the layout
    return Serverside(master), Serverside(master_diskwide), Serverside([disk,''])

# Clear all images from the layout
@bp.callback(
    Output('master', 'data'),
    Output('master-diskwide', 'data'),
    Output('last-index', 'data'),
    Input('clear', 'submit_n_clicks'),
    prevent_initial_call=True
)
def clear_all_images(n):
    if not n:
        raise PreventUpdate
    return Serverside({}), Serverside({}), Serverside(['',''])

# --------------------------- PLOTTING ---------------------------

# Enable the plot button if there are images
@bp.callback(
    Output('plot-button', 'disabled'),
    Input('master', 'data')
)
def can_plot(master):
    return not master

# Toggle the plot dialog
@bp.callback(
    Output('plot-modal', 'is_open'),
    Trigger('plot-button', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_plot():
    return True

# Change profile disk options
@bp.callback(
    Output('profile-disk-options', 'options'),
    Input('master', 'data')
)
def update_profile_disk_options(master):
    return list(master.keys())

# Change profile image (i.e. tracer) options
@bp.callback(
    Output('profile-image-options', 'options'),
    Input('profile-disk-options', 'value'),
    Input('master', 'data')
)
def update_profile_image_options(disk, master):
    if not disk or disk not in master:
        return []
    return list(master[disk].keys())

# Display plot settings once a disk and an image are given
@bp.callback(
    Output('profile-plot-settings', 'style'),
    Input('profile-disk-options', 'value'),
    Input('profile-image-options', 'value')
)
def update_profile_plot_settings(disk, images):
    return {'display': 'block'} if (disk and images) else {'display': 'none'}
    
# Update the rad and az prof checklists
@bp.callback(
    Output('rad-prof-selection', 'options'),
    Output('az-prof-selection', 'options'),
    Output('rad-label', 'children'),
    Output('az-label', 'children'),
    Input('master-diskwide', 'data'),
    Input('profile-disk-options', 'value')
)
def update_prof_selection(master_diskwide, disk):
    if not disk or disk not in master_diskwide:
        raise PreventUpdate
    rad = [{'label': f'{PA} deg', 'value': PA} for PA in master_diskwide[disk]['PA List']]
    az = [{'label': f'{rad} {master_diskwide[disk]["Unit"]}', 'value': rad} for rad in master_diskwide[disk]['Radius List']]
    rad_label = 'Radial:' if master_diskwide[disk]['PA List'] else 'Radial: No cuts entered.'
    az_label = 'Azimuthal:' if master_diskwide[disk]['Radius List'] else 'Azimuthal: No cuts entered.'
    return rad, az, rad_label, az_label

# Display radial step
@bp.callback(
    Output('rad-step-div', 'style'),
    Input('rad-prof-selection', 'options'),
    Input('subplot-selection', 'value')
)
def show_rad_step(rad, subplots):
    return {'display': 'block'} if (rad or 'rad_az' in subplots or 'avgrad' in subplots) else {'display': 'none'}
    
# Display PA Step
@bp.callback(
    Output('pa-step-div', 'style'),
    Input('rad-prof-selection', 'options'),
    Input('az-prof-selection', 'options'),
    Input('subplot-selection', 'value')
)
def show_pa_step(rad, az, subplots):
    return {'display': 'block'} if (rad or az or 'rad_az' in subplots or 'avgrad' in subplots) else {'display': 'none'}

# Display image-only options
@bp.callback(
    Output('imagery-options', 'style'),
    Input('subplot-selection', 'value')
)
def show_image_options(subplots):
    return {'display': 'block'} if ('deprj_ims' in subplots or 'or_ims' in subplots) else {'display': 'none'}

# Display au scale input field
@bp.callback(
    Output('au-scalebar', 'style'),
    Input('scalebar-switch', 'checked')
)
def show_au_scale_input(checked):
    return {'display': 'block'} if checked else {'display': 'none'}

# Display annotation switch
@bp.callback(
    Output('profile-plot-annotate-div', 'style'),
    Input('rad-prof-selection', 'value'),
    Input('az-prof-selection', 'value')
)
def show_annotation_switch(rad, az):
    return {'display': 'block'} if (rad or az) else {'display': 'none'}

# Disable csv and/or multipanel plot option if no rad or az profiles selected
@bp.callback(
    Output('output-selection', 'options'),
    Output('output-selection', 'value'),
    Output('plot-submit-alert', 'is_open'),
    Input('subplot-selection', 'value'),
    Input('rad-prof-selection', 'value'),
    Input('az-prof-selection', 'value'),
    State('output-selection', 'value')
)
def plot_selections(subplots, pa_list, rad_list, output):
    cant_multipanel = not (subplots or pa_list or rad_list)
    cant_csv = 'avgrad' not in subplots and not (pa_list or rad_list)
    if cant_multipanel and 'plot' in output:
        output.remove('plot')
    elif cant_csv and 'csv' in output:
        output.remove('csv')
    return [
        {'label': 'Multipanel plot', 'value': 'plot', 'disabled': cant_multipanel},
        {'label': 'Tabular profile data (csv)', 'value': 'csv', 'disabled': cant_csv}
    ], output, cant_multipanel

# Update the status of the profile plot submit button
@bp.callback(
    Output('profile-plot-submit-button', 'disabled'),
    Input('output-selection', 'value')
)
def update_profile_plot_submit_button(output):
    # Requirements: just select an output!
    return not output

# The big one: generate the plot!
# On downloading matplotlib figures: https://stackoverflow.com/questions/69650149/save-matplotlib-chart-from-dash-flask
# dcc.Download: https://dash.plotly.com/dash-core-components/download
@bp.callback(
    Output('download', 'data'),
    Trigger('profile-plot-submit-button', 'n_clicks'),
    State('master', 'data'),
    State('master-diskwide', 'data'),
    State('profile-disk-options', 'value'),
    State('profile-image-options', 'value'),
    State('rad-prof-selection', 'value'),
    State('az-prof-selection', 'value'),
    State('rad-step', 'value'),
    State('pa-step', 'value'),
    State('subplot-selection', 'value'),
    State('scalebar-switch', 'checked'),
    State('au-scale', 'value'),
    State('profile-plot-annotate-switch', 'checked'),
    State('output-selection', 'value'),
    prevent_initial_call=True,
    running=[
        (Output('profile-plot-submit-button', 'disabled'), True, False),
        (Output('plot-header', 'close_button'), False, True),
        (Output('profile-plot-submit-button', 'children'),
         [dbc.Spinner(size='sm'), ' Go!'],
         'Go!')
    ]
)
def generate_plot(master, master_diskwide, disk, images, pa_list, rad_list, rad_step, pa_step, subplots, make_scalebar, au_scale, draw_annotations, outputs):
    # We'll need name, image_dict, and display_dict to pass to the AstroObject
    name = master_diskwide[disk]['Name']
    
    # Crop and process the data without actually plotting it
    imdats = [master[disk][image] for image in images]
    for imdat in imdats: imdat['Data'], imdat['Scale'] = update_image(imdat['Data'], imdat, plot=False)
    # Now our image_"dict" is ready :) (it's a list now)

    # Prepare display_dict with new plot settings
    display_dict = master_diskwide[disk]
    if pa_list:
        subplots.append('rad')
    if rad_list:
        subplots.append('az')
    new_info = {
        'PA List': pa_list,
        'Radius List': rad_list,
        'Rad Step': rad_step,
        'PA Step': pa_step,
        'Plot Types': subplots,
        'AU Scale': abs(au_scale) if make_scalebar else None,
        'Annotations': draw_annotations,
        'Outputs': outputs,
    }
    for key, val in new_info.items(): display_dict[key] = val

    # Get the figure and the dataframes
    disk_object = profiles.AstroObject(name, imdats, display_dict)
    if 'plot' in outputs:
        fig = disk_object.make_plot_multiline()
    if 'csv' in outputs:
        rad = 'rad' in subplots or 'avgrad' in subplots
        az = 'az' in subplots
        if rad and az:
            dfs_list = [[(im.rad_plot_data, 'radial_profiles'), (im.az_plot_data, 'azimuthal_profiles')] for im in disk_object.imlist]
        elif rad and not az:
            dfs_list = [[(im.rad_plot_data, 'radial_profiles')] for im in disk_object.imlist]
        elif not rad and az:
            dfs_list = [[(im.az_plot_data, 'azimuthal_profiles')] for im in disk_object.imlist]

    # Moment of truth: write and download a zip file
    # https://stackoverflow.com/questions/67917360/plotly-dash-download-bytes-stream/67918580#67918580
    def write_archive(bytes_io):
        with zipfile.ZipFile(bytes_io, mode='w') as zf:
            if 'plot' in outputs:
                buf = io.BytesIO()
                fig.savefig(buf)
                img_name = f'{disk}_multipanel.png'.replace(' ', '_')
                zf.writestr(img_name, buf.getvalue())
            if 'csv' in outputs: # https://community.plotly.com/t/create-and-download-zip-file/53704/8
                for dfs, im in zip(dfs_list, disk_object.imlist):
                    for df, suffix in dfs:
                        df_bytes=df.to_csv(index=False).encode('utf-8')
                        filename = f'{disk}_{im.tag}_{suffix}.csv'.replace(' ', '_')
                        zf.writestr(filename, df_bytes)
                
    return dcc.send_bytes(write_archive, f'{disk}.zip'.replace(' ', '_'))

# --------------------------- RUN APP ---------------------------

if __name__ == '__main__':
    app = DashProxy(blueprint=bp, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
    server = app.server
    app.run_server(debug=True, use_reloader=False)