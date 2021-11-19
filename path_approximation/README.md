# Path Approximation

- Paper: Shortest path distance approximation using deep learning techniques (2018): &nbsp; [Paper](https://arxiv.org/abs/2002.05257) &nbsp;  &nbsp;   [SourceCode](https://github.com/nayash/shortest-distance-approx-deep-learning)

    + Ideas of the paper:<br>
      1\. Read data file -> graph<br>
      2\. Use Node2Vec: nodes -> embedding<br>
      3\. Create labels<br>
       &nbsp;  &nbsp;   \+ 3.1 Pick a few nodes, make them as landmark nodes (randomly)<br>
       &nbsp;  &nbsp; \+ 3.2 Calculate the distance for every pair (`l`, `n`),
        (`l` is a landmark node, `n` is a node in the graph)<br>
       4\. Create datasets with the labels from the previous step<br>
       5\. Make a model, try to predict the distance between 2 nodes<br>
       6\. Evaluate results<br>


- [Source Code from Sarel](https://github.com/kryptokommunist/path-length-approximation-deep-learning)
- [New Source Code (this repo)](https://github.com/BU-Lisp/dl-hyperbolic-random-graphs/tree/main/path_approximation)
  + Quick start:
  
    ```
    pip install -r requirements.txt
    python main.py
    ```
  + Current ideas:
    Similar to the 2018 paper, modify step 3 and 4:

    3'. Use all the nodes as landmark nodes<br>
    4'. Use all data as the training set (try to overfit the model)

