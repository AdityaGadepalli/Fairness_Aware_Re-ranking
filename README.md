# Fairness-Aware-Ranking in Search & Recommendation Systems
This implementation has been made as the final project for the course [CS60016 - AI and Ethics](http://cse.iitkgp.ac.in/~animeshm/course_aieth2020.html)

This is an implementation of the paper ["Fairness-Aware Ranking in Search & Recommendation Systems with Application to LinkedIn Talent Search"](https://dl.acm.org/doi/10.1145/3292500.3330691)

## Introduction

Ranked lists generated by recommendation systems have been becoming computationally efficient, yet there is a growing need to ensure they are free from algorithmic bias. Such biased results can lead to systematic discrimination, reduce visibility, cause over/under representation and be the reason for gender and other forms of bias. In the current implementation, such bias is quantified and mitigated via fairness-aware re-ranking algorithms. For a given search query, these algorithms can generate a desired distribution (over protected attribute(s)) of top ranked results, thereby maintaining demographic parity or equal opportunity as per requirement. Metrics are defined to assess the fairness thus achieved and results over a synthetically generated data are shown.

## Getting Started 

The [ipython notebook](https://github.com/AIEthics2020/Fairness-Aware-Ranking/blob/master/notebooks/FairnessAwareRanking_AI_Ethics_Group_13.ipynb) gives a fine overview of the paper and structurally presents the code for easy understanding.

## Project Organization
    ├── README.md                        <- Project Homepage
    |── Notebooks                        <- Sub Directory containing the Final Jupyter Notebook
    |   ──FairnessAwareRanking_AI_Ethics_Group_13.ipynb
    |── Results                          <- Sub directory to store the results
    ├── requirements.txt                 <- Python Dependencie Required to run the repo    
    ├── experiment.py                    <- The Simulation Framework 
    ├── fair_algorithms.py               <- The fair algorithms mentioned in the paper
    ├── metrics.py                       <- The set of metrics to evaluate bias
    ├── plot_utils.py                <- Utilities for plotting the results
    ├── utils.py                         <- Some utility functions
    ├── main.py                          <- The final executable to run the simulations

### Prerequisites

Once you clone the repository, please run 

```console
pip install -r requirements.txt
```

to make sure you have all the dependencies available to run the codes.

### Executing The Code

Then open your console from the same path location and run the following 

```console
python ./main.py
```

## Authors

Team Members

* [Soham Mullick](https://www.linkedin.com/in/soham-mullick-28987138/)
* [SaiKumar Korada](https://www.linkedin.com/in/saikumar-korada-565a55106/)
* [Aditya Gadepalli](https://www.linkedin.com/in/a-gadepalli/)
* [Koustav Mitra](https://www.linkedin.com/in/koustav-mitra-326105109/)
* [Deo Ashish Samanta](https://www.linkedin.com/in/deo-ashish-samanta-89a265a9/)
* [Chetan Mehatre](https://www.linkedin.com/in/chetan-mehatre-718462114/)

## Acknowledgments

The team heartfully thanks the following whose efforts and guidance have been quintessential for the current work:

* [Assoc. Prof. Animesh Mukherjee](https://cse.iitkgp.ac.in/~animeshm/) (Course Instructor)
* [Abhishek Das](https://sites.google.com/site/abhisek0193/), [Binny Mathew](https://binny-mathew.github.io/) (Teaching Assistants)
* Sahin Cem Geyik, [Stuart  Ambler](https://www.linkedin.com/in/stuart-ambler-85931a30/) , [Krishnaram Kenthapadi](https://www.linkedin.com/in/krishnaramkenthapadi/) (Authors of the paper)