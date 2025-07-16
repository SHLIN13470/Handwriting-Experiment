#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.5),
    on October 15, 2024, at 12:03
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

#########
import time

import _wintabgraphics as wintabgraphics
from _wintabgraphics import PenTracesStim 

from psychopy.gui import fileSaveDlg
from psychopy.iohub import launchHubServer
#########

import psychopy.iohub as io
from psychopy.hardware import keyboard




# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.5'
expName = 'ScriptComb'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

#########
# Default session data file name
DEFAULT_SESSION_CODE = u's1234'

# Defaults for PenPositionStim
# Pen gaussian point color when hover is detected
# PEN_POS_HOVER_COLOR = (0, 0, 255)

# Pen gaussian point color when pen press is detected
# PEN_POS_TOUCHING_COLOR = (0, 255, 0)

# Color of the pen tilt line graphic 
# PEN_POS_ANGLE_COLOR = (255, 255, 0)

# Pixel width of the pen tilt line graphic 
PEN_POS_ANGLE_WIDTH = 1

# Control the overall length of the pen tilt line graphic.
# 1.0 = default length. Set to < 1.0 to make line shorter, or > 1.0 to
# make line longer. 
PEN_POS_TILTLINE_SCALAR = 1.0

# Minimum opacity value allowed for pen position graphics.
# 0.0 = pen position disappears when pen is not detected.
PEN_POS_GFX_MIN_OPACITY = 0.0

# Minimum pen position graphic size, in normal coord space.
PEN_POS_GFX_MIN_SIZE = 0.033

# Maximum pen position graphic size, in normal coord space, is equal to
# PEN_POS_GFX_MIN_SIZE+PEN_POS_GFX_SIZE_RANGE 
PEN_POS_GFX_SIZE_RANGE = 0.033

# Defaults for PenTracesStim
# Width of pen trace line graphics (in pixels)
PEN_TRACE_LINE_WIDTH = 4

# Pen trace line color (in r,g,b 0-255)
PEN_TRACE_LINE_COLOR = (0, 0, 200)

# Pen trace line opacity. 0.0 = hidden / fully transparent, 1.0 = fully visible
PEN_TRACE_LINE_OPACITY = 1.0


# Runtime global variables
pen = None
last_evt = None
last_evt_count = 0
pen_pos_range = None
pen_data = []
#########


# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
_loggingLevel = logging.getLevel('INFO')


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Lab268_R\\Documents\\LSH\\pressure\\Old\\CalligraphyCollector_240408\\ScriptComb_1015.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp

#########
def setup_iohub_devices(expInfo, thisExp, win, sess_code=None, save_to=None):
    """
    Setup the iohub for all devices including keyboard, mouse, and pen (Wintab).
    This combines the setup for keyboard, mouse, and pen into one function, to avoid conflicts.

    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    sess_code : str or None
        The session code to use for saving the ioHub data.
    save_to : str or None
        Path to save the HDF5 data file to.

    Returns
    ==========
    ioHub server and devices (keyboard, mouse, pen).
    """
    # Create initial default session code if none provided
    if sess_code is None:
        sess_code = 'S_{0}'.format(int(time.mktime(time.localtime())))

    # Ask for session name / hdf5 file name if not provided
    if save_to is None:
        save_to = fileSaveDlg(initFilePath=os.path.dirname(__file__), initFileName=sess_code,
                              prompt="Set Session Output File",
                              allowed="ioHub Data Files (*.hdf5)|*.hdf5")
            
    if save_to:
        fdir, sess_code = os.path.split(save_to)
        sess_code = sess_code[0:min(len(sess_code), 24)]
        if sess_code.endswith('.hdf5'):
            sess_code = sess_code[:-5]
        if save_to.endswith('.hdf5'):
            save_to = save_to[:-5]
    else:
        save_to = sess_code

    # Define the iohub configuration, including the pen (Wintab), keyboard, and mouse
    exp_code = 'wintab_evts_test'
    iohub_config = {
        'experiment_code': exp_code,
        'session_code': sess_code,
        'datastore_name': save_to,
        'wintab.Wintab': {'name': 'pen',  # Pen device setup
                          'mouse_simulation': {'enable': False,
                                               'leave_region_timeout': 2.0}
                          },
        #'raw_data': True,  # Set to True to capture raw wintab data (if supported)
        'Keyboard': {'use_keymap': 'psychopy'},  # Keyboard setup
        'Mouse': {'name': 'mouse'}  # Mouse setup
    }

    # Launch ioHub with the configuration
    ioServer = launchHubServer(window=win, **iohub_config)

    # Get references to the devices
    keyboard = ioServer.devices.keyboard
    mouse = ioServer.devices.mouse
    pen = ioServer.devices.pen

    # Check that the pen device was created without any errors
    if pen.getInterfaceStatus() != "HW_OK":
        print("Error creating Wintab device:", pen.getInterfaceStatus())
        print("TABLET INIT ERROR:", pen.getLastInterfaceErrorString())
    else:
        print("Pen device initialized successfully.")
       
    # Add devices to the deviceManager for later use
    deviceManager.ioServer = ioServer
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub')
    
    if deviceManager.getDevice('Thanks_Key') is None:
        deviceManager.addDevice(deviceClass='keyboard', deviceName='Thanks_Key', backend='iohub')

    # Return the ioHub server and the devices
    return ioServer, keyboard, mouse, pen

#########

def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
#    if PILOTING:
#        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color='none', colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = 'none'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
#    if PILOTING and prefs.piloting['showPilotingIndicator']:
#        win.showPilotingIndicator()
    
    return win

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()

#########
def createPsychopyGraphics(win):
    pen_trace = wintabgraphics.PenTracesStim(win,
                                             PEN_TRACE_LINE_WIDTH,
                                             PEN_TRACE_LINE_COLOR,
                                             PEN_TRACE_LINE_OPACITY)
    pen_pos = wintabgraphics.PenPositionStim(win, PEN_POS_GFX_MIN_OPACITY,
                                             #PEN_POS_HOVER_COLOR,
                                             #PEN_POS_TOUCHING_COLOR,
                                             #PEN_POS_ANGLE_COLOR,
                                             PEN_POS_ANGLE_WIDTH,
                                             PEN_POS_GFX_MIN_SIZE,
                                             PEN_POS_GFX_SIZE_RANGE,
                                             PEN_POS_TILTLINE_SCALAR)
    return pen_trace, pen_pos
#########    
def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # Get references to the devices
    keyboard = ioServer.devices.keyboard
    mouse = ioServer.devices.mouse
    pen = ioServer.devices.pen
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    
    # Ensure pen is available
    if pen is not None:
        print("Pen device is available and initialized.")
        pen.reporting = True  # Enable pen reporting
    else:
        print("Pen device is not available!")
    
    # draw_pen_traces = True
    
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "StartExp" ---
    # Run 'Begin Experiment' code from StartExp_Code
    win.mouseVisible = False
    # Initialize pen_trace once at the start of the run function
    pen_trace = wintabgraphics.PenTracesStim(
        win, PEN_TRACE_LINE_WIDTH, PEN_TRACE_LINE_COLOR, PEN_TRACE_LINE_OPACITY)
 
    # --- Initialize components for Routine "Instructions" ---
    Instruct_Text = visual.TextStim(win=win, name='Instruct_Text',
        text='正式實驗分為兩個部分，\n兩部分之間有一段休息，總共約30分鐘。\n\n請慣用手拿觸控筆，另一隻手放在鍵盤上的空白鍵。\n\n你會看到一系列的字。當螢幕上的字快速地\n出現並消失後，請使用觸控筆在黑框內寫下\n所看到的字。\n\n當您完成該題後，請按鍵盤上的空白鍵進入下一題。\n\n如果你準備好進入練習階段，\n請使用滑鼠點擊畫面下方的 "OK"。',
        font='MS Gothic',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    Instruct_Pen = event.Mouse(win=win)
    x, y = [None, None]
    Instruct_Pen.mouseClock = core.Clock()
    Instruct_End = visual.Rect(
        win=win, name='Instruct_End',
        width=(0.12, 0.06)[0], height=(0.12, 0.06)[1],
        ori=0.0, pos=(0, -0.4), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=-2.0, interpolate=True)
    Instruct_EndText = visual.TextStim(win=win, name='Instruct_EndText',
        text='OK',
        font='Open Sans',
        pos=(0, -0.4), height=0.048, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
        
    # --- Initialize components for Routine "Blank1" ---
    Blank1_Triangle = visual.ShapeStim(
        win=win, name='Blank1_Triangle',
        size=(0.5, 0.5), vertices='triangle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=None, fillColor=None,
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "Prac" ---
    Prac_Fixation = visual.TextStim(win=win, name='Prac_Fixation',
        text='o',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    Prac_Text = visual.TextStim(win=win, name='Prac_text',
        text='',
        font='DFKai-SB',
        pos=(0, 0), height=0.2, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Prac_Stim = visual.ImageStim(
        win=win,
        name='Prac_Stim', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=(0, 0), size=(0.18, 0.18),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-1.0)
    Prac_Pen = event.Mouse(win=win)
    x, y = [None, None]
    Prac_Pen.mouseClock = core.Clock()
    Prac_Paper = visual.Rect(
        win=win, name='Prac_Paper',
        #width=(0.18, 0.18)[0], height=(0.18, 0.18)[1],
        #ori=0.0, 
        width=200, height=200,  # Dimensions in pixels
        units='pix',  # Explicitly use pixel units
        pos=(0, 0), anchor='center',
        lineWidth=5.0,     colorSpace='rgb',  lineColor='black', fillColor=None,
        opacity=None, depth=-4.0, interpolate=True)
    # Initialize graphics for pen trace and position
    pen_trace, pen_pos_gauss = createPsychopyGraphics(win)
    
    # --- Initialize components for Routine "Prac_Blank" ---
    Blank1_Triangle_2 = visual.ShapeStim(
        win=win, name='Blank1_Triangle_2',
        size=(0.5, 0.5), vertices='triangle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=None, fillColor=None,
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "Blank2" ---
    Blank2_Triangle = visual.ShapeStim(
        win=win, name='Blank2_Triangle',
        size=(0.5, 0.5), vertices='triangle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=None, fillColor=None,
        opacity=None, depth=0.0, interpolate=True)
 
    # --- Initialize components for Routine "Remind" ---
    Reminder_Text = visual.TextStim(win=win, name='Reminder_Text',
        text='練習階段已結束。\n\n如果你準備好進入主實驗，\n請按空白鍵。',
        font='MS Gothic',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    Reminder_Pen = event.Mouse(win=win)
    x, y = [None, None]
    Reminder_Pen.mouseClock = core.Clock()

        
    # --- Initialize components for Routine "Main_Break" ---
    Main_Break_Text = visual.TextStim(win=win, name='Main_Break_Text',
        text='實驗已經過半！你現在可以休息一下了。\n\n你準備好繼續的時候，\n請按空白鍵。',
        font='MS Gothic',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    Main_Break_Key = ioServer.devices.keyboard

        
    # --- Initialize components for Routine "MainExp" ---
    Main_Fixation = visual.TextStim(win=win, name='Main_Fixation',
        text='o',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    Main_Text = visual.TextStim(win=win, name='Main_text',
        text='',
        font='DFKai-SB',
        pos=(0, 0), height=0.2, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Main_Stim = visual.ImageStim(
        win=win,
        name='Main_Stim', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=(0, 0), size=(0.18, 0.18),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-1.0)
    Main_Paper = visual.Rect(
        win=win, name='Main_Paper',
        #width=(0.18, 0.18)[0], height=(0.18, 0.18)[1],
        #ori=0.0, 
        width=200, height=200,  # Dimensions in pixels
        units='pix',  # Explicitly use pixel units
        pos=(0, 0), anchor='center',
        lineWidth=5.0,     colorSpace='rgb',  lineColor='black', fillColor=None,
        opacity=None, depth=-4.0, interpolate=True)
    Main_Pen = event.Mouse(win=win)
    x, y = [None, None]
    Main_Pen.mouseClock = core.Clock()
    

    # --- Initialize components for Routine "Main_Blank" ---
    Blank1_Triangle_3 = visual.ShapeStim(
        win=win, name='Blank1_Triangle_3',
        size=(0.5, 0.5), vertices='triangle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=None, fillColor=None,
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "Thanks" ---
    Thanks_Text = visual.TextStim(win=win, name='Thanks_Text',
        text='實驗結束囉!\n\n謝謝您的參與!\n\n別忘記領取實驗參與費用。 ^_^\n\n謝謝。',
        font='MS Gothic',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    Thanks_Key = ioServer.devices.keyboard
    
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "StartExp" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('StartExp.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    StartExpComponents = []
    for thisComponent in StartExpComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "StartExp" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return

        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in StartExpComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "StartExp" ---
    for thisComponent in StartExpComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('StartExp.stopped', globalClock.getTime(format='float'))
    thisExp.nextEntry()
    # the Routine "StartExp" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Instructions.started', globalClock.getTime(format='float'))
    # setup some python lists for storing info about the Instruct_Pen

    gotValidClick = False  # until a click is received
    # Run 'Begin Routine' code from Instruct_Code
    # win.mouseVisible = False
    Instruct_Text.alignText='left'
    # keep track of which components have finished
    InstructionsComponents = [Instruct_Text, Instruct_Pen, Instruct_End, Instruct_EndText]
    for thisComponent in InstructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
            
        # *Instruct_Text* updates
        
        # if Instruct_Text is starting this frame...
        if Instruct_Text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Instruct_Text.frameNStart = frameN  # exact frame index
            Instruct_Text.tStart = t  # local t and not account for scr refresh
            Instruct_Text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Instruct_Text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Instruct_Text.started')
            # update status
            Instruct_Text.status = STARTED
            Instruct_Text.setAutoDraw(True)
        
        # if Instruct_Text is active this frame...
        if Instruct_Text.status == STARTED:
            # update params
            pass
        # *Instruct_Pen* updates
        
        # if Instruct_Pen is starting this frame...
        if Instruct_Pen.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            Instruct_Pen.frameNStart = frameN  # exact frame index
            Instruct_Pen.tStart = t  # local t and not account for scr refresh
            Instruct_Pen.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Instruct_Pen, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('Instruct_Pen.started', t)
            # update status
            Instruct_Pen.status = STARTED
            Instruct_Pen.mouseClock.reset()
            prevButtonState = Instruct_Pen.getPressed()  # if button is down already this ISN'T a new click
        
        # *Instruct_End* updates
        
        # if Instruct_End is starting this frame...
        if Instruct_End.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            Instruct_End.frameNStart = frameN  # exact frame index
            Instruct_End.tStart = t  # local t and not account for scr refresh
            Instruct_End.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Instruct_End, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Instruct_End.started')
            # update status
            Instruct_End.status = STARTED
            Instruct_End.setAutoDraw(True)
        
        # if Instruct_End is active this frame...
        if Instruct_End.status == STARTED:
            # update params
            pass
        
        # *Instruct_EndText* updates
        
        # if Instruct_EndText is starting this frame...
        if Instruct_EndText.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            Instruct_EndText.frameNStart = frameN  # exact frame index
            Instruct_EndText.tStart = t  # local t and not account for scr refresh
            Instruct_EndText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Instruct_EndText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Instruct_EndText.started')
            # update status
            Instruct_EndText.status = STARTED
            Instruct_EndText.setAutoDraw(True)
            
        # if Instruct_EndText is active this frame...
        if Instruct_EndText.status == STARTED:
            # update params
            pass
            
            
        # Run 'Each Frame' code from Instruct_Code
        if (Instruct_End.contains(Instruct_Pen) & Instruct_Pen.isPressedIn(Instruct_End)):
          continueRoutine = False
        # Check for keypress
        keys = event.getKeys(keyList=["space"])  # Replace "space" with your desired key
        if "space" in keys:
            print("Key pressed to continue!")
            continueRoutine = False  # End the current routine
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
    
            
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in InstructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
     
    
    # --- Ending Routine "Instructions" ---
    for thisComponent in InstructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Instructions.stopped', globalClock.getTime(format='float'))
    # store data for thisExp (ExperimentHandler)
    thisExp.nextEntry()
    # the Routine "Instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    

    # --- Prepare to start Routine "Blank1" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Blank1.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    Blank1Components = [Blank1_Triangle]
    for thisComponent in Blank1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Blank1" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *Blank1_Triangle* updates
        
        # if Blank1_Triangle is starting this frame...
        if Blank1_Triangle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Blank1_Triangle.frameNStart = frameN  # exact frame index
            Blank1_Triangle.tStart = t  # local t and not account for scr refresh
            Blank1_Triangle.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Blank1_Triangle, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Blank1_Triangle.started')
            # update status
            Blank1_Triangle.status = STARTED
            Blank1_Triangle.setAutoDraw(True)
        
        # if Blank1_Triangle is active this frame...
        if Blank1_Triangle.status == STARTED:
            # update params
            pass
        
        # if Blank1_Triangle is stopping this frame...
        if Blank1_Triangle.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Blank1_Triangle.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                Blank1_Triangle.tStop = t  # not accounting for scr refresh
                Blank1_Triangle.tStopRefresh = tThisFlipGlobal  # on global time
                Blank1_Triangle.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Blank1_Triangle.stopped')
                # update status
                Blank1_Triangle.status = FINISHED
                Blank1_Triangle.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Blank1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Blank1" ---
    for thisComponent in Blank1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Blank1.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.500000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    practice_loop = data.TrialHandler(nReps=1, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('CharList_P.xlsx'),
        seed=None, name='practice_loop')
    thisExp.addLoop(practice_loop)  # add the loop to the experiment
    thisPractice_loop = practice_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop.rgb)
    if thisPractice_loop != None:
        for paramName in thisPractice_loop:
            globals()[paramName] = thisPractice_loop[paramName]
    
    for thisPractice_loop in practice_loop:
        currentLoop = practice_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop.rgb)
        if thisPractice_loop != None:
            for paramName in thisPractice_loop:
                globals()[paramName] = thisPractice_loop[paramName]
        
        # --- Prepare to start Routine "Prac" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Prac.started', globalClock.getTime(format='float'))
        #Prac_Stim.setImage(Practice)
        Prac_Text.setText(PC)        
        
        vis_stim = createPsychopyGraphics(win)
        pen_trace, pen_pos_gauss = vis_stim
        # Get the current reporting / recording state of the pen
        is_reporting = pen.reporting
        # remove any events iohub has already captured.
        ioServer.clearEvents()
        # setup some python lists for storing info about the Prac_Pen
        pen_data = {'x': [], 'y': [], 'pressure': []}
        #Prac_Pen.tilt = []
        pen.time = []
        #Prac_Pen.leftButton = []
        #Prac_Pen.midButton = []
        #Prac_Pen.rightButton = []
        pen_pos_list = []
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from Prac_Code
        win.mouseVisible = False
        Prac_Paper.depth = 5.0
        pen.onPaper = []
        
        # Get x,y pen evt pos ranges for future use
        pen_pos_range = (pen.axis['x']['range'],
                         pen.axis['y']['range'])
        print(f"X-axis range: {pen.axis['x']['range']}")
        print(f"Y-axis range: {pen.axis['y']['range']}")

        # keep track of which components have finished
        PracComponents = [Prac_Fixation, Prac_Text, Prac_Paper, Prac_Pen]
        for thisComponent in PracComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Prac" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            
            # update/draw components on each frame
            
            # *Prac_Fixation* updates
            
            # if Prac_Fixation is starting this frame...
            if Prac_Fixation.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Prac_Fixation.frameNStart = frameN  # exact frame index
                Prac_Fixation.tStart = t  # local t and not account for scr refresh
                Prac_Fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Prac_Fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Prac_Fixation.started')
                # update status
                Prac_Fixation.status = STARTED
                Prac_Fixation.setAutoDraw(True)
            
            # if Prac_Fixation is active this frame...
            if Prac_Fixation.status == STARTED:
                # update params
                pass
            
            # if Prac_Fixation is stopping this frame...
            if Prac_Fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Prac_Fixation.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    Prac_Fixation.tStop = t  # not accounting for scr refresh
                    Prac_Fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    Prac_Fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Prac_Fixation.stopped')
                    # update status
                    Prac_Fixation.status = FINISHED
                    Prac_Fixation.setAutoDraw(False)
            
            # *Prac_Text* updates
            
            # if Prac_Text is starting this frame...
            if Prac_Text.status == NOT_STARTED and tThisFlip >= 1.5-frameTolerance:
                # keep track of start time/frame for later
                Prac_Text.frameNStart = frameN  # exact frame index
                Prac_Text.tStart = t  # local t and not account for scr refresh
                Prac_Text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Prac_Text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Prac_Text.started')
                # update status
                Prac_Text.status = STARTED
                Prac_Text.setAutoDraw(True)
            
            # if Prac_Text is active this frame...
            if Prac_Text.status == STARTED:
                # update params
                pass
            
            # if Prac_Text is stopping this frame...
            if Prac_Text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Prac_Text.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    Prac_Text.tStop = t  # not accounting for scr refresh
                    Prac_Text.tStopRefresh = tThisFlipGlobal  # on global time
                    Prac_Text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Prac_Text.stopped')
                    # update status
                    Prac_Text.status = FINISHED
                    Prac_Text.setAutoDraw(False)
            
            # *Prac_Paper* updates
            
            # if Prac_Paper is starting this frame...
            if Prac_Paper.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
                # keep track of start time/frame for later
                Prac_Paper.frameNStart = frameN  # exact frame index
                Prac_Paper.tStart = t  # local t and not account for scr refresh
                Prac_Paper.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Prac_Paper, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Prac_Paper.started')
                # update status
                Prac_Paper.status = STARTED
                Prac_Paper.setAutoDraw(True)
            
            # if Prac_Paper is active this frame...
            if Prac_Paper.status == STARTED:
                # update params
                pass
            
           
            # *Prac_Pen* updates
            
            # if Prac_Pen is starting this frame...
            if Prac_Pen.status == NOT_STARTED and t >= 2.6-frameTolerance:
                # keep track of start time/frame for later
                Prac_Pen.frameNStart = frameN  # exact frame index
                Prac_Pen.tStart = t  # local t and not account for scr refresh
                Prac_Pen.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Prac_Pen, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('Prac_Pen.started', t)
                # update status
                Prac_Pen.status = STARTED
                Prac_Pen.mouseClock.reset()
                prevButtonState = [0, 0, 0]  # if now button is down we will treat as 'new' click
            # Capture pen data when it's started
            if Prac_Pen.status == STARTED:  # only update if started and not finished!
               
                last_evt = None
                last_evt_count = 0
                pen_pos_range = None
                draw_pen_traces = True
                
                # Capture pen events
                # pen.reporting = is_reporting
                # Log if pen reporting is on or off
                #print(f"Pen reporting: {pen.reporting}")
                wtab_evts = pen.getSamples()
                # print(f"Captured pen events: {len(wtab_evts)}")
                # Draw each trace and position stim individually, assuming only two main components
                
                pen_trace.draw()
                pen_pos_gauss.draw()
    
                    
                if is_reporting:
                    if draw_pen_traces:
                        pen_trace.updateFromEvents(wtab_evts)

                    if last_evt_count:
                        # for e in wtab_evts:
                        #    print e
                        last_evt = wtab_evts[-1]
                        pen_pos_gauss.updateFromEvent(last_evt)
        
                else:
                    last_evt = None

                for pevt in wtab_evts:  # Iterate over each event in the list
                    #print(f"Pen Event: {pevt}")
                    # Get pen position in pixel coordinates
                    px, py = pevt.getPixPos(win)
                    #print(f"Pen Position: ({px}, {py})")
                    pressure = pevt.pressure
                    #print(f"Pen Position: ({px}, {py}), Pressure: {pressure}")
                    # Check if the pen is on paper
                    if Prac_Paper.contains((px, py)):  # Check if position is inside Prac_Paper
                        pen.onPaper.append(1)
                    else:
                        pen.onPaper.append(0)
                    # Append data to lists
                    pen_data['x'].append(px)
                    pen_data['y'].append(py)
                    pen_data['pressure'].append(pressure)
                    

                # Check for keypress
                keys = event.getKeys(keyList=["space"])  # Replace "space" with your desired key
                if "space" in keys:
                    print("Key pressed to continue!")
                    continueRoutine = False  # End the current routine
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
                

            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in PracComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Prac" ---
        for thisComponent in PracComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Prac.stopped', globalClock.getTime(format='float'))
        # store data for practice_loop (TrialHandler)
        practice_loop.addData('Prac_Pen.x', pen_data['x'])
        practice_loop.addData('Prac_Pen.y', pen_data['y'])
        practice_loop.addData('Prac_Pen.pressure', pen_data['pressure'])
        #practice_loop.addData('Prac_Pen.tilt', Prac_Pen.tilt)   
        #practice_loop.addData('Prac_Pen.leftButton', Prac_Pen.leftButton)
        #practice_loop.addData('Prac_Pen.midButton', Prac_Pen.midButton)
        #practice_loop.addData('Prac_Pen.rightButton', Prac_Pen.rightButton)
        #practice_loop.addData('Prac_Pen.time', Prac_Pen.time)
        practice_loop.addData('Prac_Pen.onPaper', pen.onPaper)
        
        # the Routine "Prac" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Prac_Blank" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Prac_Blank.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        Prac_BlankComponents = [Blank1_Triangle_2]
        for thisComponent in Prac_BlankComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Prac_Blank" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Blank1_Triangle_2* updates
            
            # if Blank1_Triangle_2 is starting this frame...
            if Blank1_Triangle_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Blank1_Triangle_2.frameNStart = frameN  # exact frame index
                Blank1_Triangle_2.tStart = t  # local t and not account for scr refresh
                Blank1_Triangle_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Blank1_Triangle_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Blank1_Triangle_2.started')
                # update status
                Blank1_Triangle_2.status = STARTED
                Blank1_Triangle_2.setAutoDraw(True)
            
            # if Blank1_Triangle_2 is active this frame...
            if Blank1_Triangle_2.status == STARTED:
                # update params
                pass
            
            # if Blank1_Triangle_2 is stopping this frame...
            if Blank1_Triangle_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Blank1_Triangle_2.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    Blank1_Triangle_2.tStop = t  # not accounting for scr refresh
                    Blank1_Triangle_2.tStopRefresh = tThisFlipGlobal  # on global time
                    Blank1_Triangle_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Blank1_Triangle_2.stopped')
                    # update status
                    Blank1_Triangle_2.status = FINISHED
                    Blank1_Triangle_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Prac_BlankComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Prac_Blank" ---
        for thisComponent in Prac_BlankComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Prac_Blank.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1 repeats of 'practice_loop'
    

    
    # --- Prepare to start Routine "Blank2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Blank2.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    Blank2Components = [Blank2_Triangle]
    for thisComponent in Blank2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Blank2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Blank2_Triangle* updates
        
        # if Blank2_Triangle is starting this frame...
        if Blank2_Triangle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Blank2_Triangle.frameNStart = frameN  # exact frame index
            Blank2_Triangle.tStart = t  # local t and not account for scr refresh
            Blank2_Triangle.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Blank2_Triangle, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Blank2_Triangle.started')
            # update status
            Blank2_Triangle.status = STARTED
            Blank2_Triangle.setAutoDraw(True)
        
        # if Blank2_Triangle is active this frame...
        if Blank2_Triangle.status == STARTED:
            # update params
            pass
        
        # if Blank2_Triangle is stopping this frame...
        if Blank2_Triangle.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Blank2_Triangle.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                Blank2_Triangle.tStop = t  # not accounting for scr refresh
                Blank2_Triangle.tStopRefresh = tThisFlipGlobal  # on global time
                Blank2_Triangle.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Blank2_Triangle.stopped')
                # update status
                Blank2_Triangle.status = FINISHED
                Blank2_Triangle.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Blank2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Blank2" ---
    for thisComponent in Blank2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Blank2.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.500000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "Remind" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Remind.started', globalClock.getTime(format='float'))
    # setup some python lists for storing info about the Reminder_Pen
    gotValidClick = False  # until a click is received
    # Run 'Begin Routine' code from Reminder_Code
    win.mouseVisible = False
    
    # keep track of which components have finished
    RemindComponents = [Reminder_Text, Reminder_Pen]
    for thisComponent in RemindComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Remind" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Reminder_Text* updates
        
        # if Reminder_Text is starting this frame...
        if Reminder_Text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Reminder_Text.frameNStart = frameN  # exact frame index
            Reminder_Text.tStart = t  # local t and not account for scr refresh
            Reminder_Text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Reminder_Text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Reminder_Text.started')
            # update status
            Reminder_Text.status = STARTED
            Reminder_Text.setAutoDraw(True)
        
        # if Reminder_Text is active this frame...
        if Reminder_Text.status == STARTED:
            # update params
            pass
        # *Reminder_Pen* updates
        
        # if Reminder_Pen is starting this frame...
        if Reminder_Pen.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            Reminder_Pen.frameNStart = frameN  # exact frame index
            Reminder_Pen.tStart = t  # local t and not account for scr refresh
            Reminder_Pen.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Reminder_Pen, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('Reminder_Pen.started', t)
            # update status
            Reminder_Pen.status = STARTED
            Reminder_Pen.mouseClock.reset()
            prevButtonState = Reminder_Pen.getPressed()  # if button is down already this ISN'T a new click

        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # Check for keypress
        keys = event.getKeys(keyList=["space"])  # Replace "space" with your desired key
        if "space" in keys:
            print("Key pressed to continue!")
            continueRoutine = False  # End the current routine
            
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in RemindComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Remind" ---
    for thisComponent in RemindComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Remind.stopped', globalClock.getTime(format='float'))
    # store data for thisExp (ExperimentHandler)
    thisExp.nextEntry()
    # the Routine "Remind" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    main_loop = data.TrialHandler(nReps=1, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('CharList_C.xlsx', selection='0:150'),
        seed=None, name='main_loop')
    thisExp.addLoop(main_loop)  # add the loop to the experiment
    thisMain_loop = main_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisMain_loop.rgb)
    if thisMain_loop != None:
        for paramName in thisMain_loop:
            globals()[paramName] = thisMain_loop[paramName]
    
    for thisMain_loop in main_loop:
        currentLoop = main_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # Add the break functionality
        #if main_loop.thisN == 2:  # Check if trial number is 2
            # Display the break message and wait for a key press
        #    continueRoutine = True
        #    while continueRoutine:  # Keep displaying the message until a key is pressed
        #        Main_Break_Text.draw()
        #        win.flip()

                # Check for keypress
        #        keys = event.getKeys(keyList=["space"])  # Replace "space" with your desired key
        #        if "space" in keys:
        #            print("Key pressed to continue!")
        #            continueRoutine = False  # Exit the loop to continue the experiment

        
        # abbreviate parameter names if possible (e.g. rgb = thisMain_loop.rgb)
        if thisMain_loop != None:
            for paramName in thisMain_loop:
                globals()[paramName] = thisMain_loop[paramName]
        
        # --- Prepare to start Routine "MainExp" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('MainExp.started', globalClock.getTime(format='float'))
        #Main_Stim.setImage(Main)
        Main_Text.setText(SC_MC)  
       
        vis_stim = createPsychopyGraphics(win)
        pen_trace, pen_pos_gauss = vis_stim
        # Get the current reporting / recording state of the pen
        is_reporting = pen.reporting
        # remove any events iohub has already captured.
        ioServer.clearEvents()
        # setup some python lists for storing info about the Main_Pen
        pen_data = {'x': [], 'y': [], 'pressure': []}
        #Main_Pen.leftButton = []
        #Main_Pen.midButton = []
        #Main_Pen.rightButton = []
        Main_Pen.time = []
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from Main_Code
        win.mouseVisible = False
        Main_Paper.depth = 1.0
        Main_Pen.onPaper = []
                
        # Get x,y pen evt pos ranges for future use
        pen_pos_range = (pen.axis['x']['range'],
                         pen.axis['y']['range'])
        #print(f"X-axis range: {pen.axis['x']['range']}")
        #print(f"Y-axis range: {pen.axis['y']['range']}")
        
        
        # keep track of which components have finished
        MainExpComponents = [Main_Fixation, Main_Text, Main_Paper, Main_Pen]
        for thisComponent in MainExpComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "MainExp" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Main_Fixation* updates
            
            # if Main_Fixation is starting this frame...
            if Main_Fixation.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Main_Fixation.frameNStart = frameN  # exact frame index
                Main_Fixation.tStart = t  # local t and not account for scr refresh
                Main_Fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Main_Fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Main_Fixation.started')
                # update status
                Main_Fixation.status = STARTED
                Main_Fixation.setAutoDraw(True)
            
            # if Main_Fixation is active this frame...
            if Main_Fixation.status == STARTED:
                # update params
                pass
            
            # if Main_Fixation is stopping this frame...
            if Main_Fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Main_Fixation.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    Main_Fixation.tStop = t  # not accounting for scr refresh
                    Main_Fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    Main_Fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Main_Fixation.stopped')
                    # update status
                    Main_Fixation.status = FINISHED
                    Main_Fixation.setAutoDraw(False)
            
            # *Main_Text* updates
            
            # if Main_Text is starting this frame...
            if Main_Text.status == NOT_STARTED and tThisFlip >= 1.5-frameTolerance:
                # keep track of start time/frame for later
                Main_Text.frameNStart = frameN  # exact frame index
                Main_Text.tStart = t  # local t and not account for scr refresh
                Main_Text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Main_Text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Main_Text.started')
                # update status
                Main_Text.status = STARTED
                Main_Text.setAutoDraw(True)
            
            # if Main_Text is active this frame...
            if Main_Text.status == STARTED:
                # update params
                pass
            
            # if Main_Text is stopping this frame...
            if Main_Text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Main_Text.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    Main_Text.tStop = t  # not accounting for scr refresh
                    Main_Text.tStopRefresh = tThisFlipGlobal  # on global time
                    Main_Text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Main_Text.stopped')
                    # update status
                    Main_Text.status = FINISHED
                    Main_Text.setAutoDraw(False)
            
            # *Main_Paper* updates
            
            # if Main_Paper is starting this frame...
            if Main_Paper.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
                # keep track of start time/frame for later
                Main_Paper.frameNStart = frameN  # exact frame index
                Main_Paper.tStart = t  # local t and not account for scr refresh
                Main_Paper.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Main_Paper, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Main_Paper.started')
                # update status
                Main_Paper.status = STARTED
                Main_Paper.setAutoDraw(True)
            
            # if Main_Paper is active this frame...
            if Main_Paper.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from Main_Code
            win.mouseVisible = False
           
           # if Main_Pen is starting this frame...
            if Main_Pen.status == NOT_STARTED and t >= 2.6-frameTolerance:
                # keep track of start time/frame for later
                Main_Pen.frameNStart = frameN  # exact frame index
                Main_Pen.tStart = t  # local t and not account for scr refresh
                Main_Pen.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Main_Pen, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('Main_Pen.started', t)
                # update status
                Main_Pen.status = STARTED
                Main_Pen.mouseClock.reset()
                prevButtonState = [0, 0, 0]  # if now button is down we will treat as 'new' click
                
            # Capture pen data when it's started
            if Main_Pen.status == STARTED:  # only update if started and not finished!
               
                last_evt = None
                last_evt_count = 0
                pen_pos_range = None
                draw_pen_traces = True 
                
                # Capture pen events
                # pen.reporting = is_reporting
                wtab_evts = pen.getSamples()
                
                # Draw each trace and position stim individually, assuming only two main components
                pen_trace.draw()
                pen_pos_gauss.draw()
                    
                if is_reporting:
                    if draw_pen_traces:
                        pen_trace.updateFromEvents(wtab_evts)

                    if last_evt_count:
                        # for e in wtab_evts:
                        #    print e
                        last_evt = wtab_evts[-1]

                        pen_pos_gauss.updateFromEvent(last_evt)

                        
                else:
                    last_evt = None

                for pevt in wtab_evts:  # Iterate over each event in the list
                    #print(f"Pen Event: {pevt}")
                    # Get pen position in pixel coordinates
                    px, py = pevt.getPixPos(win)
                    #print(f"Pen Position: ({px}, {py})")
                    pressure = pevt.pressure
                    #print(f"Pen Position: ({px}, {py}), Pressure: {pressure}")
                    # Check if the pen is on paper
                    if Main_Paper.contains((px, py)):  # Check if position is inside Main_Paper
                        pen.onPaper.append(1)
                    else:
                        pen.onPaper.append(0)
                    # Append data to lists
                    pen_data['x'].append(px)
                    pen_data['y'].append(py)
                    pen_data['pressure'].append(pressure)   
                    
                # Check for keypress
                keys = event.getKeys(keyList=["space"])  # Replace "space" with your desired key
                if "space" in keys:
                    print("Key pressed to continue!")
                    continueRoutine = False  # End the current routine
                    
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in MainExpComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "MainExp" ---
        for thisComponent in MainExpComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('MainExp.stopped', globalClock.getTime(format='float'))
        # store data for main_loop (TrialHandler)
        
        main_loop.addData('Main_Pen.x', pen_data['x'])
        main_loop.addData('Main_Pen.y', pen_data['y'])
        main_loop.addData('Main_Pen.pressure', pen_data['pressure'])
        #main_loop.addData('Main_Pen.leftButton', Main_Pen.leftButton)
        #main_loop.addData('Main_Pen.midButton', Main_Pen.midButton)
        #main_loop.addData('Main_Pen.rightButton', Main_Pen.rightButton)
        #main_loop.addData('Main_Pen.time', Main_Pen.time)
        # Run 'End Routine' code from Main_Code
        main_loop.addData('Main_Pen.onPaper', pen.onPaper)
        
        
        
        # the Routine "MainExp" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Main_Blank" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Main_Blank.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        Main_BlankComponents = [Blank1_Triangle_3]
        for thisComponent in Main_BlankComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Main_Blank" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Blank1_Triangle_3* updates
            
            # if Blank1_Triangle_3 is starting this frame...
            if Blank1_Triangle_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Blank1_Triangle_3.frameNStart = frameN  # exact frame index
                Blank1_Triangle_3.tStart = t  # local t and not account for scr refresh
                Blank1_Triangle_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Blank1_Triangle_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Blank1_Triangle_3.started')
                # update status
                Blank1_Triangle_3.status = STARTED
                Blank1_Triangle_3.setAutoDraw(True)
            
            # if Blank1_Triangle_3 is active this frame...
            if Blank1_Triangle_3.status == STARTED:
                # update params
                pass
            
            # if Blank1_Triangle_3 is stopping this frame...
            if Blank1_Triangle_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Blank1_Triangle_3.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    Blank1_Triangle_3.tStop = t  # not accounting for scr refresh
                    Blank1_Triangle_3.tStopRefresh = tThisFlipGlobal  # on global time
                    Blank1_Triangle_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Blank1_Triangle_3.stopped')
                    # update status
                    Blank1_Triangle_3.status = FINISHED
                    Blank1_Triangle_3.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Main_BlankComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Main_Blank" ---
        for thisComponent in Main_BlankComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Main_Blank.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1 repeats of 'main_loop'
    
    
    # --- Prepare to start Routine "Thanks" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Thanks.started', globalClock.getTime(format='float'))

    # Initialize a list to track keypresses
    _Thanks_Key_allKeys = []

    # keep track of which components have finished
    ThanksComponents = [Thanks_Text]  # Exclude Thanks_Key as it's now handled separately
    for thisComponent in ThanksComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1

    # --- Run Routine "Thanks" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)

        # update/draw components on each frame

        # *Thanks_Text* updates
        if Thanks_Text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Thanks_Text.frameNStart = frameN  # exact frame index
            Thanks_Text.tStart = t  # local t and not account for scr refresh
            Thanks_Text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Thanks_Text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Thanks_Text.started')
            Thanks_Text.status = STARTED
            Thanks_Text.setAutoDraw(True)

        # Check for keypresses using iohub keyboard
        keypresses = Thanks_Key.getPresses(keys=['q', 'escape'])  # Example: check for 'q' or 'escape' key presses
        if keypresses:
            for key in keypresses:
                if key == 'q':
                    continueRoutine = False  # End the routine if 'q' is pressed
                if key == 'escape':
                    thisExp.status = FINISHED  # Quit the experiment if 'escape' is pressed
                    continueRoutine = False
                    core.quit()  # Exit the experiment

        # Check if the experiment should end
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            break

        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component is still running
        for thisComponent in ThanksComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished

        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()

    # --- Ending Routine "Thanks" ---
    for thisComponent in ThanksComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Thanks.stopped', globalClock.getTime(format='float'))

    # check responses
    if len(_Thanks_Key_allKeys) == 0:  # No response was made
        Thanks_Key.keys = None
    else:
        # Log the last key press details (optional, for tracking key presses)
        last_key = _Thanks_Key_allKeys[-1]
        thisExp.addData('Thanks_Key.keys', last_key.name)
        thisExp.addData('Thanks_Key.rt', last_key.rt)

    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED

    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    #expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    
    save_to = fileSaveDlg(initFilePath=os.path.dirname(__file__), initFileName=DEFAULT_SESSION_CODE,
                          prompt="Set Session Output File",
                          allowed="ioHub Data Files (*.hdf5)|*.hdf5")
                          
    win = setupWindow(expInfo=expInfo)
    

    #Setup iohub devices (keyboard, mouse, pen) and get references to the devices
    ioServer, keyboard, mouse, pen = setup_iohub_devices(expInfo, thisExp, win)
    
    print ("Axis: ", pen.axis)
    print ("context: ", pen.context)
    print ("model: ", pen.model)
    
    # Check the status of the pen device
    device_status = pen.getInterfaceStatus()
    print(f"Pen device interface status: {device_status}") 

    
    # Start the experiment
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
        )
        
    
    # Save the experiment data
    saveData(thisExp=thisExp)
        
    # Quit the experiment
    quit(thisExp=thisExp, win=win)
