# Utilities

### generate_dirs

Basic Utilities for the Python Scripts

### runbg.sh

This code runs python scripts in the background.

You can make it a command to everyone by moving it to /usr/bin by,

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