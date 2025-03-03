# Utilities

### file_utils.py 

Utilities for generating files, timing functions, etc.

### runbg.sh

This code runs python scripts in the background.

You can make it a command to everyone by moving it to /usr/bin. Here is the code to do so,

```bash
sudo mv utils/runbg.sh /usr/bin/runbg
```

Use the l flag to generate a log file.

Debugging:

If you are getting a permission denied error:
Make sure to run 

```bash
chmod +x /usr/bin/runbg
``` 

### subset_data.py

Subsets data for when the dataset is too large process (e.g. takes too long to run).