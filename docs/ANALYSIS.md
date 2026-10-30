# Test-set error analysis

This report was generated from the held-out test split after model selection and validation threshold tuning. The test labels were used only for this final evaluation and the error list below.

- Selected model: `tfidf_unigram_bigram_stop_removed_MultinomialNB`
- Validation threshold: `0.298678228883` (precision target `0.98`; constraint met: `True`)
- Test spam precision: `1.000000`; recall: `0.877551`; F1: `0.934783`
- Confusion matrix (actual rows ham/spam, predicted columns ham/spam): `[[678, 0], [12, 86]]`

## False positives (all held-out test errors)

None on the held-out test split.

## False negatives (all held-out test errors)

- Actual `spam`, predicted `ham`, spam probability `0.187369`: Would you like to see my XXX pics they are so hot they were nearly banned in the uk!. Likely reason: The wording or formatting differs from the older training examples and did not cross the tuned threshold.
- Actual `spam`, predicted `ham`, spam probability `0.145950`: Burger King - Wanna play footy at a top stadium? Get 2 Burger King before 1st Sept and go Large or Super with Coca-Cola and walk out a winner. Likely reason: The wording or formatting differs from the older training examples and did not cross the tuned threshold.
- Actual `spam`, predicted `ham`, spam probability `0.295916`: Thanks for the Vote. Now sing along with the stars with Karaoke on your mobile. For a FREE link just reply with SING now.. Likely reason: The wording or formatting differs from the older training examples and did not cross the tuned threshold.
- Actual `spam`, predicted `ham`, spam probability `0.076337`: RCT' THNQ Adrian for U text. Rgds Vatian. Likely reason: The wording or formatting differs from the older training examples and did not cross the tuned threshold.
- Actual `spam`, predicted `ham`, spam probability `0.253017`: CLAIRE here am havin borin time & am now alone U wanna cum over 2nite? Chat now <PHONE> hope 2 C U Luv CLAIRE xx Calls£1/minmoremobsEMSPOBox45PO139WA. Likely reason: The wording or formatting differs from the older training examples and did not cross the tuned threshold.
- Actual `spam`, predicted `ham`, spam probability `0.234243`: Latest News! Police station toilet stolen, cops have nothing to go on!. Likely reason: The wording or formatting differs from the older training examples and did not cross the tuned threshold.
- Actual `spam`, predicted `ham`, spam probability `0.039217`: Check Out Choose Your Babe Videos @ <URL> fgkslpoPW fgkslpo. Likely reason: The wording or formatting differs from the older training examples and did not cross the tuned threshold.
- Actual `spam`, predicted `ham`, spam probability `0.264832`: Hi its LUCY Hubby at meetins all day Fri & I will B alone at hotel U fancy cumin over? Pls leave msg 2day <PHONE> Lucy x Calls£1/minMobsmoreLKPOBOX177HP51FL. Likely reason: The wording or formatting differs from the older training examples and did not cross the tuned threshold.
- Actual `spam`, predicted `ham`, spam probability `0.006202`: Babe: U want me dont u baby! Im nasty and have a thing 4 filthyguys. Fancy a rude time with a sexy bitch. How about we go slo n hard! Txt XXX SLO(4msgs). Likely reason: The wording or formatting differs from the older training examples and did not cross the tuned threshold.
- Actual `spam`, predicted `ham`, spam probability `0.112788`: Hi babe its Jordan, how r u? Im home from abroad and lonely, text me back if u wanna chat xxSP <URL> Text stop to stopCost 150p <PHONE>. Likely reason: The wording or formatting differs from the older training examples and did not cross the tuned threshold.
- Actual `spam`, predicted `ham`, spam probability `0.027579`: Oh my god! I've found your number again! I'm so glad, text me back xafter this msgs cst std ntwk chg £1.50. Likely reason: The wording or formatting differs from the older training examples and did not cross the tuned threshold.
- Actual `spam`, predicted `ham`, spam probability `0.158796`: Your credits have been topped up for <URL> Your renewal Pin is tgxxrz. Likely reason: The wording or formatting differs from the older training examples and did not cross the tuned threshold.

## Indicative features

These are learned model weights, not causal explanations.

| More indicative of spam | Weight | More indicative of ham | Weight |
|---|---:|---|---:|
| `<phone>` | 3.79926 | `gt` | 3.56135 |
| `<num> p` | 3.52513 | `lt` | 3.55637 |
| `claim` | 3.26366 | `lor` | 3.45993 |
| `prize` | 3.25463 | `lt gt` | 3.45167 |
| `won` | 3.03159 | `ok` | 3.26364 |
| `<phone> <num>` | 3.02874 | `come` | 3.24473 |
| `nokia` | 2.83872 | `home` | 3.21812 |
| `guaranteed` | 2.71808 | `i'll` | 3.20741 |
| `box <num>` | 2.71226 | `i'm` | 3.15021 |
| `<curr> prize` | 2.69282 | `da` | 3.14861 |
| `<curr> cash` | 2.65355 | `oh` | 3.04796 |
| `<curr>` | 2.55412 | `going` | 2.98565 |
| `ppm` | 2.54237 | `later` | 2.97519 |
| `tone` | 2.53476 | `doing` | 2.87965 |
| `ringtone` | 2.50039 | `got` | 2.85689 |

## Message length statistics after deduplication

| Class | Messages | Mean chars | Median chars | Min | Max | Mean preprocessed tokens |
|---|---:|---:|---:|---:|---:|---:|
| ham | 4518 | 70.89 | 53.0 | 2 | 910 | 14.31 |
| spam | 653 | 137.71 | 148.0 | 13 | 223 | 26.13 |

## Modern SMS examples (DEMO ONLY)

These hand-written examples are illustrative only. They are never included in training, validation, test metrics, or model selection.

| Scenario | Expected demo label | Model prediction | Spam probability | Redacted message |
|---|---|---|---:|---|
| UPI account suspension scam | spam | ham | 0.1263 | Your UPI account is blocked. Share the OTP now to avoid suspension. |
| KYC renewal link scam | spam | ham | 0.2083 | KYC expired today. Verify immediately at <URL> to keep your wallet active. |
| UPI refund fee scam | spam | spam | 0.3320 | UPI refund pending. Pay Rs 2 verification fee to receive Rs 10000 now. |
| lottery prize scam | spam | spam | 0.9028 | Congratulations! You won a lottery prize. Claim now by replying with your bank details. |
| delivery redelivery scam | spam | ham | 0.0231 | Package held. Pay a redelivery fee at <URL> today. |
| ordinary meeting update | ham | ham | 0.0378 | Meeting moved to 3 pm. See you at the office. |
| ordinary grocery reminder | ham | ham | 0.0039 | Can you pick up milk on your way home? |
| ordinary account safety notice | ham | spam | 0.6097 | Your verification code is 000000. Never share this code with anyone. |

## Obfuscation checks (DEMO ONLY)

These adversarial-style examples are separate from the test set and are not evaluation metrics.

| Variant | Prediction | Spam probability | Redacted example |
|---|---|---:|---|
| plain baseline | spam | 0.9052 | Congratulations! You won a free prize, click now |
| punctuated spelling | spam | 0.4865 | Congratulations! You won a f.r.e.e prize, click now |
| digit substitution | spam | 0.7748 | Congratulations! You won a FR33 prize, click now |
| spaced letters | spam | 0.4865 | Congratulations! You won a f r e e prize, click now |
| Cyrillic homoglyphs | ham | 0.1362 | Cоngratulations! Yоu wоn a free prize, click now |
| zero-width character | spam | 0.7859 | Congratulations! Youwon a free prize, click now |

Interpretation: zero-width characters are stripped before feature extraction, but punctuation, digit substitutions, spacing, and non-Latin homoglyphs can still change the token sequence. The observed predictions above are the only claims made about these examples.

## Limitations

The UCI corpus is old and drawn from English-language SMS sources, including UK and Singapore contexts. Its vocabulary, carriers, scam patterns, and class balance do not represent current messaging traffic. In particular, modern OTP/KYC/UPI and delivery scams in India can be out of domain. A high validation precision target does not guarantee the same precision in production. The model should be treated as a review aid, and messages near the decision boundary should not be automatically blocked without a human or policy layer.
