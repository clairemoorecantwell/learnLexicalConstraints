# Online learner with lexically specific constraints

## Quick Start

To quickly get started learning, you need a couple of things

* Some data, formatted properly
  * first line, tab-separated: `input` `output`...
  * `tab.prob` and (hidden structure) are optional, just leave those columns out if your don't want to use them
  * input column should have something on every line: each line is a candidate, and each the input column should contain the UR for that candidate.  Each UR needs to be unique.
  * The violation columns can contain any integers, or spaces.  Spaces will be converted to 0's and integers will be converted to negative integers.
* Some way to run Python (2.7), ideally that can plot, so that the plotting functions will work.  I use ipython <https://ipython.org>
* A plan!



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

**candidates:** A list containing objects of the class `candidate`. This has to be populated after the UR object is created.

**probDenom:** This is for calculating MaxEnt probability for candidates of this input.  It's just the denominator for that calculation: sum(e^(-H))

**predProbsList:** List of each candidate's predicted probability, `predictedProb`, for use in sampling

**obsProbList:** List of each candidate's observed probability, `observedProb`, for use in sampling

**lastSeen:** For use in online learning, this is a tag that notes when the UR was last sampled.  It's used for decaying the lexically-specific constraints.

**nSeen:** For use in online learning, this is a tag that notes how many times the UR has been sampled.  For now, it's just for use in analyzing the learning process after the fact - it's not used during learning

**probableCandidates:** a list of all the candidates with observed probabilities greater than 5%, or 0.05

`UR` also has the following methods:

`addCandidate(cand)` adds a candidate object to the list of candidates

`getProbableCandidates()` fills *probableCandidates* with the surface forms of all candidates whith observed probabilities greater than 5%.  This can be seen to approximate a Bayesian "Highest Density Interval" or "Credible Interval" of output candidates.

`decayLexC(t,decayRate,decayType='linear')` Decays the lexical constraints according to how much time has passed since the UR was last observed.  The `decayRate` is a parameter defining how much they decay by.  If it's 0, they won't decay at all.  The lexical constraint weights will decay according to the equation: *new_weight = old_weight - (t-`lastSeen`) X decayRate*, and if the weight decays to or below zero, the lexical constraint will get removed. Note that there's a parameter `decayType` for which the only supported value is currently `linear`.  That's because there's probably a better way to decay, but a previously implemented `logarithmic` decay actually made no sense, and is now commented out.  So this is work in progress here, but I've left a slot for the future work to go in.

`checkViolationLength()` checks if all the violation vectors of the candidates are the same length ** Not written yet**

`calculateHarmony(w,t=None,decayRate=None,decayType=None)` calculates the harmony of each candidate, using w, the vector of weights of the markedness constraints.  If `decayRate` is set to something besides None, it begins by calling `decayLexC`, which is why it takes `t`, `decayRate`, and `decayType` as parameters.

`predictProbs(w,t=None,decayRate=None,decayType=None)` calculates the predicted probabilities of each candidate according to a vector of weights `w`.  Begins by calling `calculateHarmony`, which can call `decayLexC`, hence you can give it the appropriate parameters

`getPredWinner(theory)` Get a winner based on your theory - if `MaxEnt` (currently the only one implemented), sample to get that winner.  Returns a string that's the actual surface form winner, as well as the entire candidate object that's the winner.

`getObsWinner(theory)` Get an observed winner by sampling from the observed distribution over candidates.  If there's a sole observed winner, that one will always be returned by this function.  Returns a string that's the surface form of the observed winner, and also returns the entire candidate object.

`compareObsPred(theory,w,strategy = 'sample', t=None,decayRate=None,decayType=None)` Compares observed to predicted winner, using one of two strategies.  Either samples from both the predicted and the observed distributions, and compares those samples to see if there's an error, or if `strategy` = `HDI`, it samples from the predicted distribution, and then checks whether that sampled form is one of the most probable candidates for in the observed distribution.

If `sample`, the default, it calls `getPredWinner` and `getObsWinner`, and compares the observed to the predicted winner to see if there's an error.  The function also calls `predictProbs`, which calls `calculateHarmony` and `decayLexC`, so altogether, it calculates the current values for the lexical constraints, and it calculates harmonies and probabilities given the current set of both general and lexical constraint weights, and then it samples and sees if the observed thing matches the predicted thing.  Returns whether or not there was an error (0 if no error, 1 if error), and the candidate objects for the observed candidate and the predicted candidate.  NOTE ON HIDDEN STRUCTURE:  **hidden structure still in progress.  check back later ** 



## Tableaux

use: `new_tableaux = Tableaux(theory='MaxEnt')`

attributes:

**urList:** list of objects of class `UR`

**urIDlist:** list of strings, that are the strings of the URs

**urProbsList:** List of floats, aligned to the list of URs - each number is the 'lexical frequency' of that UR, or the weight at which you want it to be sampled for learning.  These can by any real positive number (or 0) - for sampling the list will be converted to a proper probability distribution.

**theory:** Right now, only 'MaxEnt' is implemented.  Hopefully in the future, we can do at least Noisy HG and regular HG.

**constraints:** List of names of the markedness/faithfulness constraints

**w:** List of weights of the markedness/faithfulness constraints

**t:** Current or most recent learning iteration

**lexicon:** This keeps track of what words have actually been presented to the learner, and how often.  Its structure is [[],[]], or a list containing two sublists.  The first is a list of UR objects, and the second is a list of their corresponding integer 'frequencies', which indicate the number of times that UR has been used for learning.  For example, [[ur1,ur2,ur3],[12,45,0]] indicates that ur1 has been used 12 times, ur2 has been used 45 times, and ur3 has not been used at all.

**pLexC:** The probability that, upon error, a lexical constraint will be induced.  This starts out as 1, and can get lowered over the course of learning if you're using probabilistic induction of these constraints.



_methods:_

`addUR(ur)` adds an object of class `UR` to the `Tableaux`

`initializeWeights(w)` Takes an optional argument, `w`, which is the weight vector you want to use for starting weights.  If no `w` is provided, the weights are all initialized to zero.

`getReady()` Creates a new attribute for the Tableaux object: URsampleVector, which is a list of probabilities that's normalized from urProbsList, and turned into a numpy array if numpy is turned on.

`sample()` Uses `URsampleVector` to sample a UR, and returns that UR object.  This is no longer actually used inside the learning functions, but you might find it useful for playing around with single iterations or troubleshooting.

`resetTime()` Resets the time stamps of all the parts of the tableaux object - the `t` value, as well as the `lastSeen` value of each UR.

`resetLexC()` Goes through the URs and erases all their lexical constraints.

`downSample(size,disregardFrequency=False)` returns a new Tableaux object with only `size` UR's, sampled without replacement from the original Tableaux object.  By default, it samples them according to their given frequencies, but you can turn this off by setting `disregardFrequency` to `True`

`update(theory, learnRate, lexCstartW, lexLearnRate, lexCSample=False, lexCSampleSize=10, decayRate=None, decayType=None, haveLexC=False, comparisonStrategy='sample', urToUse=None)` Executes a single update iteration.  See the paper/presentation for a schematic of how this iteration proceeds.  This function returns the UR object updated on, and a 1 or 0, indicating whether there was an error or not (1 means error, 0 means no error).  Inside the function the following things happen:

  - Check if a UR has been provided in `urToUse`.  If not, get one using `sample()`.
  - Add the UR being used to the `lexicon`, or up its count if it's already in the `lexicon`.
  - use `compareObsPred()` to decide whether there is an error. `comparisonStrategy` is a parameter that decided whether to compare via plain sampling (`sample`), or compare the predicted output to a set of 'credible outputs' in the observed distribution (`HDI`).  See the documentation of `compareObsPred()` for details.
  - If there's an error:
    ..- use `perceptronUpdate()` to update all the general constraints
    ..- if any lexical constraints exist for this UR, update them according to perceptron update rule.  NOTE: a hard upper limit of 700 is imposed on lexical constraint weights.  This is because around 740 or so, exponentiating the negative to calculate the probability yields a zero, which later yields a division by zero error
    ..- if there's no lexical constraint for this UR that prefers the correct output, induce one
    ..- (if lexical constraints are being induced probabilistically, first do some calculations to decide what the right probability is)
    ..-((This sampling process is still under construction))
  - increment time `t`
  - update the trained UR's `lastSeen` and `nSeen` parameters

  Ok, now let's talk about all the parameters:
  .. theory
  .. learnRate
  .. lexCstartW
  .. lexLearnRate
  .. lexCSample
  .. lexCSampleSize
  .. decayRate
  .. decayType
  .. haveLexC
  .. comparisonStrategy
  .. urToUse

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


* Think about sensible 'noisy' options

* Keep track of how long learning takes - may need to think about how to optimize!

* Make `initializeWeights()` run automatically at some sensible point

* Add to `initializeWeights()`:
  * Ability for user to specify a list of initial weights
  * Ability to choose random weights, with user-specified parameters

* Figure out how to implement regularization



#### Visualization:

* Make graphs within Python
  * figure out how to visualize weights of lexical constraints
  	* add functionality to the tableau object, for it to take groupings as input
  	* hover/click capability
  * histogram of UR's chosen by sampling
  * SSE/MSE/logLikelihood over time

* Make export capability for reporting in R

## Wish list:

* Implement option for different learning rates for each general constraint, a la Jesney and Tessier

* Implement batch gradient descent option
  * Add lexC generation
  * Figure out what to do with hidden structure

* Add stress candidate generation
