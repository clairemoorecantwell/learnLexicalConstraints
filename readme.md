# Online learner with lexically specific constraints

## Classes:

### candidate
use: `new_candidate = candidate(c,violations,observedProb,surfaceForm=None)`

**c:** the candidate

**violations:** list of violations

**surfaceForm:** for use in hidden structure situations, this is the surface form that corresponds to the actual candidate.  For example, a stress candidate might be (10)0, and its surface form 100.  The surface form for (1)00, which has a different violation profile, would also be 100.

**observedProb:** Value ranging from 0-1 indicating the observed probability of that candidate.  If this candidate is the sole observed output for its UR, this value should be 1.  If the candidate is never observed, this should be 0.  Except in hidden structure situations, these values for any given UR ought to sum to one.  Also, this is the value that whatever learning process will try to match.

**predictedProb:** The predicted probability of that candidate given the current constraint weights and the theory you're using (e.g. MaxEnt) As of 25 Mar, 2016, only MaxEnt is implemented.  NOTE: This is calculated on the fly during learning.  If you're learning via sampling, this value will be whatever it was calculated to be last time predictProbs() was called on the parent UR.

**harmony:** The harmony of the candidate, calculates by calling calculateHarmony() from the parent UR

`candidate` also has the following methods:

`checkViolationsSign()` runs automatically on initialization of a new instance of the class, and converts violation profiles to negative floats.  It also converts blanks to 0's, but it will throw an error if any violations are made of text that can't be converted to floats.

### UR
use: `new_UR = UR(ur,prob=1,lexC=None)`

**ur:** a string, the UR, or the input if you prefer (Note: it's called UR because 'input' is a function in python, not because of any theoretical commitments.)

**prob:** you can use this attribute to define a token frequency for this UR, for example if you want to sample or weight URs/inputs by token frequency

**lexC:** This starts out empty, and can stay empty if you're not using lexically-specific constraints.  It should typically not be defined in advance, but you can if you need to.  It should wind up being a list of tuples, of the form `(output, weight)` where `output` is a string corresponding to the output that the lexically-specific constraint demands, and `weight` is the weight on that constraint.  __Lexical constraints are hitched to URs/inputs, and are not stored with the other constraints.  This might need to be changed to allow for batch learning __

`UR` also has the following attributes:

**candidates:** A list containing objects of the class `candidate`

**probDenom:** This is for calculating MaxEnt probability for candidates of this input.  It's just the denominator for that calculation: sum(e^(-H))

**predProbsList:** List of each candidate's predicted probability, `predictedProb`, for use in sampling

**obsProbList:** List of each candidates observed probability, `observedProb`, for use in sampling

**lastSeen:** For use in online learning, this is a tag that notes when the UR was last sampled.  It's used for decaying the lexically-specific constraints

`UR` also has the following methods:

`addCandidate(cand)` adds a candidate to the list of candidates

`decayLexC(t,decayRate,decayType='linear')` Decays the lexical constraints according to how much time has passed since the UR was observed.  The `decayRate` is a parameter defining how much they decay by.  If it's 0, they won't decay at all.  `decayType` can be `linear` or `logarithmic`.  If `linear`, the lexical constraint weights will decay according to the equation: *new_weight = old_weight - (t-`lastSeen`) X decayRate*, and if the weight decays to or below zero, the lexical constraint will get removed. If decay is set to `logarithmic`, the weights will decay according to *new_weight = old_weight - (t-lastSeen)^decayRate*.  Once again, if it gets to or below zero the constraint will get removed.

`checkViolationLength()` checks if all the violation vectors of the candidates are the same length ** Not written yet**

`calculateHarmony(w,t=None,decayRate=None,decayType=None)` calculates the harmony of each candidate, using w, the vector of weights of the markedness constraints.  If `decayRate` is set to something besides None, it begins by calling `decayLexC`, which is why it takes `t`, `decayRate`, and `decayType` as parameters.

`predictProbs(w,t=None,decayRate=None,decayType=None)` calculates the predicted probabilities of each candidate according to a vector of weights `w`.  Begins by calling `calculateHarmony`, which can call `decayLexC`, hence you can give it the appropriate parameters

`getPredWinner(theory)` Get a winner based on your theory - if MaxEnt (currently the only one implemented), sample to get that winner.  Returns a string that's the actual surface form winner, as well as the entire candidate object that's the winner.

`getObsWinner(theory)` Get an observed winner by sampling from the observed distribution over candidates.  If there's a sole observed winner, that one will always be returned by this function.  Returns a string that's the surface form of the observed winner, and also returns the entire candidate object.

`compareObsPred(theory,w,t=None,decayRate=None,decayType=None)` Calls `getPredWinner` and `getObsWinner`, and compares the observed to the predicted winner to see if there's an error.  The function also calls `predictProbs`, which calls `calculateHarmony` and `decayLexC`, so altogether, it calculates the current values for the lexical constraints, and it calculates harmonies and probabilities given the current set of both general and lexical constraint weights, and then it samples and sees if the observed thing matches the predicted thing.  Returns whether or not there was an error (0 if no error, 1 if error), and the candidate objects for the observed candidate and the predicted candidate.  NOTE ON HIDDEN STRUCTURE:  **hidden structure still in progress.  check back later ** 

## Tableaux

use: `new_tableaux = Tableaux(theory='MaxEnt')`

attributes:

**urList:** list of objects of class `UR`

**urIDlist:** list of strings, that are the strings of the URs

**urProbsList:** List of floats, aligned to the list of URs - each number is the 'lexical frequency' of that UR, or the weight at which you want it to be sampled for learning

**theory:** Right now, only 'MaxEnt' is implemented

**constraints:** List of names of the markedness/faithfulness constraints

**w:** List of weights of the markedness/faithfulness constraints

**t:** Current or last learning iteration

methods:

`addUR(ur)` adds an object of class `UR` to the `Tableaux`

`initializeWeights()` Initializes the weights to zero  *NOTE: need to fix this so that it can take arbitrary or random starting weights*

`sample()` Uses `urProbsList` to sample a UR (for use in updating)

`update(theory, learnRate, lexCstartW, decayRate=None, decayType=None)` Executes a single update iteration.  The steps of this are (1) sample a UR using `sample()`, (2) compare the predicted output given the current grammar to the observed outpus using `compareObsPred()`.  (3) If there's an error, first (3a) update the general constraints with `perceptronUpdate`, then (3b) update the lexC's.  If there's already a lexC favoring the observed form, increment it by the learning rate.  If there's any lexC's favoring a different surface form, decrement them by the learning rate.  If there's no lexC favoring the observed surface form, create one with weight *lexCstartW*  (4) Lastly, increment *t*.  The function returns the UR that was used, and whether or not there was an error (1 for error, 0 for no error)

`epoch(theory, iterations, learnRate, lexCstartW, decayRate=None, decayType=None)` Runs `update()` *iterations* number of times.  Returns the error rate during the epoch, and two lists, one of lexCs and one of weights of those lexCs

`learn(iterations, nEpochs, learnRate, lexCstartW, decayRate=None, decayType='linear', theory='MaxEnt')` Runs *nEpochs* epochs, each with *iterations* number of iterations.

`printTableau()` Prints the current tableaux to the command line

`calcLikelihood()` Calculate log likelihood for the entire data set *NOT YET IMPLEMENTED*

`resetTime()` Resets derivational time to zero, for the whole tableau, and for each UR as well



## Functions:

`readOTSoft(file)` Right now, this just has the capability to read in an hgr-style file and turn it into a Tableaux object, and to yell at you if the file type is wrong.

`perceptronUpdate(error, target, weights, rate)` error and target should be violation vectors, weights should be a vector of weights, and rate is the learning rate.  Returns an updated weight vector.


## To-Do list:

* fix `printTableau()` so that it does something more visually appealing
  * Have it print the first n lines of tableaux, in case your tableaux are really big

* fix the carriage return problem... Ideally, accept both \r and \n as line splits

* Take away annoying print functions

* Think about sensible 'noisy' options

* Keep track of how long learning takes - may need to think about how to optimize!

* Make `initializeWeights()` run automatically at some sensible point

* Add to `initializeWeights()`:
  * Ability for user to specify a list of initial weights
  * Ability to choose random weights, with user-specified parameters

* Figure out how to implement regularization



#### Visualization:

* Make graphs within Python
  * weights of general constraints over time
  * figure out how to visualize weights of lexical constraints
  * histogram of UR's chosen by sampling
  * SSE/MSE/logLikelihood over time

* Make export capability for reporting in R

## Wish list:

* Implement option for different learning rates for each general constraint, a la Jesney and Tessier

* Implement batch gradient descent option
  * Add lexC generation
  * Figure out what to do with hidden structure

* Add stress candidate generation
