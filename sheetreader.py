"""
Converts any google sheet workbook into a dictionary compatible with figg.process().
---
Original code written by Alex DelFranco
Adapted by Bibi Hanselman
Original dated 10 July 2024
Updated 14 July 2024
"""

import pandas as pd
import glob

######################

def get_sheet_data(wb,name):
  '''
  Input: A google spreadsheet workbook and the name of the workbook sheet
  Output: A dataframe of the data contained on the given sheet
  Description: Pull data from a google sheet into a dataframe
  '''
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
  '''
  Input: The name of a google spreadsheet workbook and the preferred mockup number
  Output: A dictionary of data from that spreadsheet
  Description: Extracts and returns a dictionary of data from a given google sheet
  '''
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

def get_list(colname,wb,settings_sheet):
    """
    Converts a single sheets column into a list with omitted header and empty values
    
    Parameters
    ----------
    wb : ~gspread.spreadsheet.Spreadsheet
        Google Sheets workbook instance, pulled from the desired url.
    settings_sheet : str
        Name of settings sheet from which to pull column.

    Returns
    -------
    dir_list : list
        List of column values.

    """
    if colname in get_sheet_data(wb,settings_sheet):
        col_list = get_sheet_data(wb,settings_sheet)[colname].tolist()
        while '' in col_list: col_list.remove('')
    else:
        # If there's no hierarchy given, just return an empty list so it searches
        # the root directory
        col_list = []
    return col_list

######################

def global_setup(wb,settings_sheet):
    '''
    '''
    # Pull relevant dictionaries
    settings_dict = sheet_extract(wb,settings_sheet)
    defaults, specs = {}, {}
  
    # Add default information to a dictionary
    for index,default in enumerate(settings_dict['Input']):
        # Don't add if an default isn't specified
        if default == '': continue
        # Add enable switches and input values to the dictionary
        defaults[settings_dict['Input'][index]] = settings_dict['Value'][index]
    
    # Add subdirectory specifications to a dictionary
    for index,subdir in enumerate(settings_dict['Subdirectory']):
        # Don't add if a subdir isn't specified
        if subdir == '': continue
        # Add all spec values as one entry in the dictionary
        specs[subdir] = {}
        for spec in settings_dict['Specs']:
            # Don't add if an spec isn't specified
            if spec == '': continue
            specs[subdir][spec] = settings_dict[spec][index]

    # Return the defaults dictionary
    return defaults, specs

######################

def addpaths(wb,im_data,namekey,settings_sheet):
    """
    Adds paths for every image in the dataset by searching for the inputted
    file name. If settings_sheet is None, searches the root directory. Otherwise,
    searches in subdirectories determined by designated image parameters.

    Parameters
    ----------
    wb : ~gspread.spreadsheet.Spreadsheet
        Google Sheets workbook instance, pulled from the desired Sheets interface.
    im_data : dict
        Image dictionary, containing parameter values for a single image.
    settings_sheet : str
        Name of settings sheet from which to pull default values. If not None,
        must include 'Hierarchy' column, which lists the image parameters,
        in hierarchical order, to draw upon to construct the file path.

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
        # Get the list of ordered columns from which to obtain,
        # for each image, the subdirectory at each hierarchical tier.
        lvls = get_list('Hierarchy',wb,settings_sheet)
        
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

def get_subdir(mockup,index,wb,settings_sheet):
    """
    """
    subdir=''
    for lvl in get_list("Hierarchy", wb, settings_sheet):
        try:
            temp = mockup[lvl][index] + '/'
        except KeyError:
            print('Your data sheet does not contain the parameter ' + lvl +
                  'needed to create the subdirectory!')
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
  wb : ~gspread.spreadsheet.Spreadsheet
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
      defaults,specs = global_setup(wb,settings_sheet)
  else: defaults,specs = {},{}

  # Add image-specific information to a main dictionary
  images = {}
  
  # Check if dicts should be split by a splitkey. If yes, conceive the children.
  if splitkey is not None:
      splitkeys = list(set(mockup[splitkey]))
      
      # Initialize the child dicts
      for key in splitkeys:
          images[key] = {}
          
  for index,objname in enumerate(mockup[namekey]):
    # For each image, loop through all the possible data inputs
    image = {}
    
    # Get subdirectory for the image
    if settings_sheet is not None: 
        subdir = get_subdir(mockup, index, wb, settings_sheet)
    
    for key in mockup:
      # If there is already an input, enter it in the dictionary
      if mockup[key][index] != '':
        image[key] = mockup[key][index]
      # If there isn't, check if it was given as a global parameter
      elif key in defaults:
        if defaults[key] != '': image[key] = defaults[key]
      # If it's not global, check if it's a subdirectory spec
      # This is not elegant. Return to later. 7/14/24
      elif settings_sheet is not None:
        subdir = get_subdir(mockup, index, wb, settings_sheet)
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
                  image[key] = image[key].split(',')
      
      # Convert checkbox 'TRUE' values to 'True' (bool)
      if image[key] == 'TRUE': image[key] = True
      
      # Finally, convert empty strings to NoneTypes
      if image[key] == '': image[key] = None
      
    # If add_paths is True, add paths to the image dictionary
    if add_paths: addpaths(wb,image,namekey,settings_sheet)
    
    # Add the image dictionary to a master dictionary
    if splitkey is None:
        images[objname] = image
    else: images[image[splitkey]][objname] = image
    
  # Return the master dictionary
  return images