## Results

The system has been tested on sample video inputs and is able to detect suspicious and theft-related activities in real time.

Below is an example of the current output:

* The system identifies human presence using pose estimation
* Temporal behavior is analyzed using the trained LSTM model
* When suspicious activity is detected, an alert is triggered along with a confidence score

Example output includes:

* Bounding box around the detected person
* Alert message indicating theft detection
* Confidence score displayed on the dashboard

These results demonstrate that the overall pipeline - from video input to final prediction - is working as expected.

At the current stage, the system performs reliably on controlled test cases. However, further improvements are ongoing to enhance accuracy, reduce false positives, and make the system more robust for real-world scenarios.

Future improvements include:

* Testing on larger and more diverse datasets
* Improving model accuracy and generalization
* Optimizing real-time performance (FPS)
* Enhancing the frontend dashboard for better visualization

