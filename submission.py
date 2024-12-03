import pandas as pd

sample_submission = pd.read_csv('sample_submission.csv')
sample_submission['answer'] = no_pad_output
sample_submission.to_csv('submission.csv', index=False)
