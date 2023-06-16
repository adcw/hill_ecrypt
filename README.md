# hill_cipher
Encoding, decoding using Hill method and Cipher breaking

Autors:
Przemysław Kawa, Adrian Ćwiąkała

The classic shotgun method with a few modifications was used to crack the cipher
- the climb method generates the inverse of the key, so the resulting key is its inverse.
- climbing is done on multiple processes. Every 1000 iterations, 
  processes consolidate to change keys for better ones or duplicate keys to another process.
- the percentage of key change decreases as the quality of the text decrypted with it increases, so that at the very beginning
  the key is changed extensively, and only slightly at the end.
- If we're getting close to a solution, a function is run to evaluate which rows need to be changed to break the key.
   This method is based on calculating the quality using the bigram method and evaluating which letters of the decrypted text are already correct,
   and which need to be changed. Only those rows that generate letters that do not match the others are changed.
- Results of manual tests in English:
     The minimum length of a text with a 2x2 key for which the algorithm will always do the work in less than 30s is 290
     Average time and percentage of successful solutions when the key length is unknown and the text length is 3336
     keys not resolved within the specified time period are marked as not resolved

| key_len  | mean_timer | success rate | min/max_recorded | time_limit |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 2  | 3.35  | 100%  | 0.03/19.49  | 30s  |
| 3  | 165.6  | 98.5%  | 127.2/226.3  | 1 hour  |
| 4  | 720.4  | 82%  | 495.8/1312.3(21min)  | 1 hour  |

NOTE: if we do not include the result of 1312, the average will be 644.6 

- NOTE for above ken_len 2: It is worth noting that the process of closing and opening sub-processes takes moments before and after completion
 		on my computer it takes about 20s in total (I noticed different times on colleagues).
 		Additionally, the method tries to resolve ken_len 2 every time (when the key length is not known)
 		for 30s, this causes the results to be inflated for a total of about 50s+-10s.
		So e.g. the result of 132s ken_len 3 in the test is 71s of solving key 3 alone.
- The test platform is: amd ryzen 9 3900xt
