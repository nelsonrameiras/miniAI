## Description
Brief description of the changes in this PR.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue).
- [ ] New feature (non-breaking change which adds functionality).
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected).
- [ ] Documentation update.
- [ ] Performance improvement.
- [ ] Code refactoring.
- [ ] CI/CD improvement.

## Related Issues
Fixes #(issue number).
Related to #(issue number).

## Changes Made
- Change 1.
- Change 2.
- Change 3.

## Testing Performed
- [ ] Compiled successfully.
- [ ] Ran all existing tests.
- [ ] Added new tests for new features.
- [ ] Tested on Linux.
- [ ] Tested on macOS.
- [ ] Ran benchmark to check performance impact.

### Test Commands
```bash
# Commands used to test
# pex.
make clean && make
./miniAI test --model IO/models/digit_brain.bin --dataset digits --static
```

### Test Results
```
Paste relevant test output here
```

## Documentation
- [ ] Updated README.md.
- [ ] Updated inline code comments.
- [ ] Added/updated examples.

## Code Quality
- [ ] Code follows the project's style guidelines.
- [ ] Self-review of code completed.
- [ ] No new compiler warnings.
- [ ] No memory leaks (tested with valgrind).
- [ ] Maintains zero-dependency philosophy.

## Screenshots (if applicable)
Add screenshots or terminal output showing the changes.

## Performance Impact
- [ ] No performance impact.
- [ ] Performance improved.
- [ ] Performance degraded (explain why acceptable below).

**Details:**

## Breaking Changes
If this introduces breaking changes, describe:
- What breaks.
- Migration path for users.
- Version bump needed (major/minor/patch).

## Additional Notes
Any additional information that reviewers should know.

## Checklist
- [ ] My code follows the style guidelines of this project.
- [ ] I have performed a self-review of my own code.
- [ ] I have commented my code, particularly in hard-to-understand areas.
- [ ] I have made corresponding changes to the documentation.
- [ ] My changes generate no new warnings.
- [ ] I have added tests that prove my fix is effective or that my feature works.
- [ ] New and existing unit tests pass locally with my changes.
- [ ] Any dependent changes have been merged and published.