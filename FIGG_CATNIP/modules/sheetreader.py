"""
Converts any google sheet workbook into a dictionary compatible with figg.process().
---
Original code written by Alex DelFranco
Adapted by Bibi Hanselman
Original dated 10 July 2024
Updated 30 July 2024
"""

import pandas as pd
import glob

######################

def get_sheet_data(wb,name):
    """
    Pulls data from a google sheet into a dataframe.
    
    Parameters
    ----------
    wb : gspread.spreadsheet.Spreadsheet
        Google Sheets workbook instance, pulled from the desired url.
    name : str
        Name of the desired sheet in the workbook from which to pull data.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe containing column data from the given sheet.

    """
    # Get data from the spreadsheet
    sheet = wb.worksheet(name)
    data = sheet.get_all_values()
    df = pd.DataFrame(data)
  
    # Arrange Pandas dataframe
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    df = df.reset_index()
  
    # Return the dataframe
    return df

######################

def sheet_extract(wb,sheet,trim=False):
    """
    Extracts and returns a dictionary of data from a given google sheet.

    Parameters
    ----------
    wb : gspread.spreadsheet.Spreadsheet
        Google Sheets workbook instance, pulled from the desired url.
    sheet : str
        Name of the desired sheet in the workbook from which to pull data.
    trim : bool, optional
        Option to trim empty values. The default is False.

    Returns
    -------
    data : dict
        Dictionary in which each column name is associated with column values
        stored in lists.

    """
    # Get the data from the google sheet
    dataframe = get_sheet_data(wb,sheet)
    
    # Create a dictionary of dataframe column names with simple references
    col_names = dataframe.columns.values.tolist()[1:]
    
    # For each of the columns of data add the data to the dictionary
    data = {}
    for column in col_names:
        # Covert each column to a list
        coldat = dataframe[column].values.tolist()
        # Trim empty values
        while trim and '' in coldat: coldat.remove('')
        # Add the column to the dictionary of data
        data[column] = coldat
    
    # Return the dictionary
    return data

######################

def global_setup(wb,settings_sheet):
    """
    Pulls global default values and custom subdirectory specs from a given
    settings sheet.

    Parameters
    ----------
    wb : gspread.spreadsheet.Spreadsheet
        Google Sheets workbook instance, pulled from the desired url.
    settings_sheet : str
        Name of settings sheet from which to pull default values. To populate 
        each dict, your sheet must contain the following columns:
            For defaults:
            'Input': Parameters to which to assign a default value.
            'Value': Corresponding default values for the 'Input' parameters.
            For specs:
            'Subdirectory': Subdirectories for which custom specs are desired.
            'Specs': list of columns, each representing a parameter, containing
                default values for the corresponding subdirectories in the
                'Subdirectory' column. Must include additional columns whose
                names are the values in this column so that these columns can
                be referenced by the code.

    Returns
    -------
    defaults : dict
        Dictionary that stores global defaults in parameter:value pairs.
    specs : dict
        Dictionary that stores custom specifications for each subdirectory in 
        nested dictionaries.

    """
    # Pull relevant dictionaries
    settings_dict = sheet_extract(wb,settings_sheet)
    defaults, specs = {}, {}
  
    # Add default information to a dictionary, if applicable
    if 'Input' in settings_dict and 'Value' in settings_dict:
        for index,default in enumerate(settings_dict['Input']):
            # Don't add if an default isn't specified
            if default == '': continue
            # Add enable switches and input values to the dictionary
            defaults[settings_dict['Input'][index]] = settings_dict['Value'][index]
        
    # Add subdirectory specifications to a dictionary, if applicable
    if 'Subdirectory' in settings_dict and 'Specs' in settings_dict:
        for index,subdir in enumerate(settings_dict['Subdirectory']):
            # Don't add if a subdir isn't specified
            if subdir == '': continue
            # Add all spec values as one entry in the dictionary
            specs[subdir] = {}
            for spec in settings_dict['Specs']:
                # Don't add if an spec isn't specified
                if spec == '': continue
                try:
                    specs[subdir][spec] = settings_dict[spec][index]
                except NameError:
                    print('Cannot pull spec values for ' + spec + 
                        'because the column ' + spec + 'does not exist in your settings sheet!')

    # Return the dicts
    return defaults, specs

######################

def addpaths(wb,im_data,namekey,settings_sheet):
    """
    Adds paths for every image in the dataset by searching for the inputted
    file name. If settings_sheet is None, searches the root directory. Otherwise,
    searches in subdirectories determined by designated image parameters.

    Parameters
    ----------
    wb : gspread.spreadsheet.Spreadsheet
        Google Sheets workbook instance, pulled from the desired Sheets interface.
    im_data : dict
        Image dictionary, containing parameter values for a single image.
    settings_sheet : dict
        Dictionary of data pulled from a settings sheet.
    Returns
    -------
    im_data : dict
        Expanded dictionary with added key 'Path' for each image.
    """
    if settings_sheet is None:
        if 'File Name' in im_data:
            for path in glob.glob('/*.fits*'):
                if im_data['File Name'] in path:
                    im_data['Path'] = path
                    break
                
            # If no path was found, throw an error
            if 'Path' not in im_data:
                raise FileNotFoundError("Drats! No file was found for " + im_data[namekey]
                                        + " in the root directory. Please double check your directory or sheet!")
        else:
            raise KeyError('No file name given for ' + im_data[namekey] + 
                           '. Please check if your sheet contains the required column "File Name".')
    else:
        if im_data['Subdirectory Override'] is not None: filedir = im_data['Subdirectory Override']
        else:
            # Get the list of ordered columns from which to obtain
            # the subdirectory at each hierarchical tier.
            lvls = settings_sheet['Hierarchy']
            
            # Declare file directory string
            filedir = ''
            
            # Create the directory string based on the values in the image dictionary
            for lvl in lvls:
                # Check if the subdirectory column has a value - skip if it doesn't
                if im_data[lvl] == None: continue
                
                # Add the subdirectory to the path
                temp = im_data[lvl] + '/'
                filedir += temp
        
        # Get all paths in that directory
        paths = glob.glob(filedir+'*.fit*')
        
        # Find the file path we're looking for!
        if 'File Name' in im_data:
            for path in paths:
                if im_data['File Name'] in path:
                    im_data['Path'] = path
                    break
            
            # If no path was found, throw an error
            if 'Path' not in im_data:
                raise FileNotFoundError("Drats! No file was found for " + im_data[namekey]
                                        + " in the directory " + filedir +
                                        ". Please check your directory!")
                
        else:
            raise KeyError("No file name given for " + im_data[namekey] + 
                           ". Please check if your sheet contains the required column 'File Name'.")

    # Return the expanded image dictionary
    return im_data
    
######################

def get_subdir(mockup,index,settings_sheet):
    """
    During the assignment of parameter values to an image dictionary, 
    obtains subdirectory location of the file path for a given image index.

    Parameters
    ----------
    mockup : dict
        Dictionary of data from the image data sheet.
    index : int
        Index of desired image in the lists stored in the mockup dict.
    wb : gspread.spreadsheet.Spreadsheet
        Google Sheets workbook instance, pulled from the desired Sheets interface.
    settings_sheet : dict
        Dictionary of data pulled from a settings sheet. MUST contain 'Hierarchy' list
        of main sheet columns to pull from to generate subdirectory path.

    Returns
    -------
    subdir : str
        Subdirectory of the desired image.
    """
    subdir=''
    for lvl in settings_sheet['Hierarchy']:
        try:
            temp = mockup[lvl][index] + '/'
        except KeyError:
            print(f'Your data sheet does not contain the parameter {lvl} needed to create the subdirectory!')
        subdir += temp
    return subdir
    
######################

def wb_to_dict(wb,sheet,add_paths=False,settings_sheet=None,namekey='Object',splitkey=None):
  """
  Creates a data dictionary out of a given spreadsheet.
  The keys of this dictionary are the object names, which must be given by 
  values in a column titled 'Object.' Each value is a dictionary containing
  parameter values for one such object (row), where each key is the name of a
  column (parameter).

  Parameters
  ----------
  wb : gspread.spreadsheet.Spreadsheet
      Google Sheets workbook instance, pulled from the desired url.
  sheet : str
      Name of desired sheet within the workbook.
  add_paths : bool, optional
      Choose whether to add paths to the final master dictionary using glob 
      (path finder). Set to True if file paths are not directly given in 
      sheet under a column titled 'Path,' but file names or fragments thereof
      are given under a column titled 'File Name.' The default is False.
  settings_sheet : str, optional
      Name of settings sheet from which to pull default values.
      The default is None.
  namekey : str, optional
      Name of column from which to assign keys to inner (image) dicts.
      The default is 'Object'.
  splitkey : str, optional
      Name of column whose values are to divide image dicts into larger child dicts.
      The default is None.

  Returns
  -------
  imdat : dict
      Dictionary containing parameter values for all images.
      Each image has a nested dictionary of parameter values.
  """
  # Pull mockup data from the sheet
  mockup = sheet_extract(wb,sheet)
  
  # Pull general settings from the sheet, if such a sheet is given
  if settings_sheet is not None:
      settings = sheet_extract(wb, settings_sheet, trim=True)
      defaults,specs = global_setup(wb,settings_sheet)
  else: defaults,specs = {},{}

  # Add image-specific information to a main dictionary
  images = {}
  
  # Check if dicts should be split by a splitkey. If yes, make subdicts
  if splitkey is not None:
      splitkeys = list(set(mockup[splitkey]))
      
      # Initialize the subdicts
      for key in splitkeys:
          images[key] = {}
          
  for index,objname in enumerate(mockup[namekey]):
    # For each image, loop through all the possible data inputs
    image = {}
    
    for key in mockup:
      # If there is already an input, enter it in the dictionary
      if mockup[key][index] != '':
        image[key] = mockup[key][index]
      # If there isn't, check if it was given as a global parameter
      elif key in defaults:
        if defaults[key] != '': image[key] = defaults[key]
      # If it's not global, check if it's a subdirectory spec
      # This is not a nice solution. Return to later, maybe. 7/14/24
      elif settings_sheet is not None and 'Hierarchy' in settings:
        subdir = get_subdir(mockup, index, settings) if mockup['Subdirectory Override'][index] == '' else mockup['Subdirectory Override'][index]
        if subdir in specs:
          if key in specs[subdir]:
            image[key] = specs[subdir][key]
          else: image[key] = ''
      # Otherwise, assign an empty string
      else: image[key] = ''
    
    for key in image:
      # Change the numeric entries to numbers
      # Check if the string could be a float
      if all([i.isnumeric() for i in image[key].split('.',1)]):
        # Change it to a float
        image[key] = float(image[key])
        # If the float could be an integer, change it
        if image[key].is_integer(): image[key] = int(image[key])
      
      # Handle splitting of lists/ranges preemptively
      if isinstance(image[key], str):   
          for sym in [',',':']:
              if sym in image[key]:
                  image[key] = image[key].split(sym)
      
      # Convert checkbox values to bools
      if image[key] == 'TRUE': image[key] = True
      elif image[key] == 'FALSE': image[key] = False
      
      # Finally, convert empty strings to NoneTypes
      if image[key] == '': image[key] = None
      
    # If add_paths is True, add paths to the image dictionary
    if add_paths: addpaths(wb,image,namekey,settings)
    
    # Add the image dictionary to a master dictionary
    if splitkey is None:
        images[objname] = image
    else: images[image[splitkey]][objname] = image
    
  # Return the master dictionary
  return images