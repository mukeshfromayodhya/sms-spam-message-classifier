# Dataset

`SMSSpamCollection` is the extracted tab-separated text file from the official UCI SMS Spam Collection archive. Source: https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection. The dataset is attributed to Almeida and Hidalgo (2011), DOI https://doi.org/10.24432/C5CC84, and UCI lists it under CC BY 4.0.

`sms-spam-collection.zip` is the downloaded source archive. The training loader reads the extracted `SMSSpamCollection` file, removes exact duplicate labeled rows, and only then creates the stratified splits.
