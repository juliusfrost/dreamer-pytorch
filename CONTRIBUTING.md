# Contributing

## Project
Use the repository project page to check what needs to be done. Then assign yourself to the project card. 
Create new cards for fine grain implementation details.

## Commiting
Always work on new branches and not directly on the master branch.
Name them appropriately to the feature you are implementing.
Submit a pull request to merge your completed feature into the master branch.
Don't mix multiple features into your pull requests. 

## Testing

Always try to test your code before you push to a branch. 
Tests are required to pass when merging a pull request into the master branch.

To run tests:
```bash
pytest tests
```

If you want additional code coverage information:
```bash
pytest tests --cov=dreamer
```

### Styling

Use PEP8 style for python syntax. (ctrl-alt-l in PyCharm) 
You can check the status of PEP8 linting in your pull requests too.
