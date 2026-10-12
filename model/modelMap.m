

modelWRAPPER is the only place to do anything 
    modelPath = 'path to /model';
    subjNum = 17;
    appendStr = 'subject_17_session_4'; 
    constrain = 'Thresh0'; 

    *get all the data from the subject you want, and consider the constraint (only take files matching it)* 
    

    If no training data or model exists, creates both.

    If new files are found:
    - Linear model: incrementally trains on the new data.
    - CNN model: retrains from scratch using all current source files.

    If existing files are changed or removed, retrains from scratch.

    If nothing has changed, skips training.
    

that's it for here 

then in the app, push the button and get this message: 

AVAILABLE MODELS
1 | model_kilosort_NS17

Enter model number: 