# Online learner with phonological form constraints

## Quick Start

To quickly get started learning, you need a couple of things

* Some data, formatted properly

|input | output | probability | tab.prob | Align-R | Align-L|...|
|------|--------|-------------|----------|---------|--------|---|
|banana| 010 | 1 | 547 | 0 | 1|
|banana| 100 | 0 | 547 | 1 | 0 |
|recipe| 010 | 0 | 388 | 0 | 1 |
|recipe| 100 | 1 | 388 | 1 | 0 |
| ... | ... | ...| ... | ... | ... |

  * first line, tab-separated: `input` `output`...
  * `tab.prob` and (hidden structure) are optional, just leave those columns out if your don't want to use them
  * input column should have something on every line: each line is a candidate, and each the input column should contain the input for that candidate.  Each input needs to be unique.  If you run the model without PFC's, as a simple MaxEnt gradual learner, this would be the UR.
  * The violation columns can contain any integers, or spaces.  Spaces will be converted to 0's and integers will be converted to negative integers.
* Some way to run Python (Version 3), ideally that can plot, so that the plotting functions will work.  I use ipython <https://ipython.org>.  If you can install numpy (<http://www.numpy.org/>), and do it in a way that you get access to the numpy.random.choice() function, the whole thing will run much quicker.  You might need to install some kind of package like Anaconda (<https://www.continuum.io/downloads>) to get the numpy installation to run smoothly.

* You can start off with the two example files 'comparativesToy.txt' and 'comparativesCOCA.txt'.  The former contains 100 inputs, with frequencies uniformly distributed, of 1-100, and all with 50% probability of taking each comparative.  Only two 'phonological' constraints are in there, and they simply prefer one or the other of the comparative forms.  'comparativesCOCA.txt' contains 214 adjectives of English, with frequencies and probabilities obtained from COCA, and 9 grammatical constraints with violations.  For more details about how these tableaux were constructed, please visit <http://http://www.phrenology.biz/emergentIdiosyncrasy/>.

* Within python, you'll need to import the GLaPL module, read in your tableaux file with `readOTSoft()`, initialize weights if you wish, and then use the `Tableaux.learn()` function to do some learning.  You should save the results of your learning to a results object (demonstrated below), and then use `Results.save()` to save a bunch of text files that you can then import to R or whatever program you wish for further inspection.

Here's a demonstration:

`import GLaPL`

`comparatives = GLaPL.readOTSoft('comparativesCOCA.txt')`

`comparatives.initializeWeights([1,1,1,1,1,1,1,1,1])` _note:_ skip this step if you want your weights to start at 0.

`comp_results = comparatives.learn(1000, 100, 0.01, PFCstartW=10, pfcLearnRate=0.1, PFCSample=True, PFCSampleSize=20, decayRate=0.0001, decayType='static', havePFC=True)` 100,000 iterations, split into 100 epochs. Learning rate is 0.01.  Phonological Form Constraints (PFCs) are induced with an initial weight of 10, updated at a rate of 0.1, and decay at a rate of 0.0001 at each timestep.  Note that the default behavior of the learn() function is to do plain perceptron learning without any phonological form constraints.  To get the PFCs, you have to set havePFC to True.  Also, decayType has a few settings, but 'static' is the most straightforward (and definitely not buggy).

`comp_results.save('myFirstLearningResults')`  This will create a bunch of files prefixed with 'myFirstLearningResults', in the same directory as this module.  For example, 'myFirstLearningResults_weights.txt' will contain weights of all the general constraints at the end of each epoch.

## Classes:

### candidate
use: `new_candidate = candidate(c,violations,observedProb,surfaceForm=None)`

**c:** the candidate

**violations:** list of violations

**surfaceForm:** for use in (future!) hidden structure situations, this is the surface form that corresponds to the actual candidate.  For example, a stress candidate might be (10)0, and its surface form 100.  The surface form for (1)00, which has a different violation profile, would also be 100.

**observedProb:** Value ranging from 0-1 indicating the observed probability of that candidate.  If this candidate is the sole observed output for its UR, this value should be 1.  If the candidate is never observed, this should be 0.  Except in hidden structure situations, these values for any given UR ought to sum to one.  Also, this is the value that whatever learning process will try to match.

**predictedProb:** The predicted probability of that candidate given the current constraint weights and the theory you're using (e.g. MaxEnt) As of 25 Mar, 2016, only MaxEnt is implemented.  NOTE: This is calculated on the fly during learning.  If you're learning via sampling, this value will be whatever it was calculated to be last time predictProbs() was called on the parent UR.

**harmony:** The harmony of the candidate, calculated by calling calculateHarmony() from the parent UR

`candidate` also has the following methods:

`checkViolationsSign()` runs automatically on initialization of a new instance of the class, and converts violation profiles to negative floats.  It also converts blanks to 0's, but it will throw an error if any violations are made of text that can't be converted to floats.

### LexEntry
use: `new_LexEntry = LexEntry(inpt,prob=1,pfc=None)`

**inpt:** a string, the input, or the UR if you like

**prob:** you can use this attribute to define a token frequency for this input, for example if you want to sample or weight inputs by token frequency

**pfc:** This starts out empty, and can stay empty if you're not using PFCs.  It should typically not be defined in advance, but you can if you need to.  It should wind up being a list of tuples, of the form `(output, weight)` where `output` is a string corresponding to the output that the PFC demands, and `weight` is the weight on that constraint.  __PFCs are hitched to inputs, and are not stored with the other constraints.  This might need to be changed to allow for batch learning __

`LexEntry` also has the following attributes:

**candidates:** A list containing objects of the class `candidate`. This has to be populated after the LexEntry object is created.

**probDenom:** This is for calculating MaxEnt probability for candidates of this input.  It's just the denominator for that calculation: sum(e^(-H))

**predProbsList:** List of each candidate's predicted probability, `predictedProb`, for use in sampling

**obsProbList:** List of each candidate's observed probability, `observedProb`, for use in sampling

**lastSeen:** For use in online learning, this is a tag that notes when the UR was last sampled.  It's used for decaying the lexically-specific constraints.

**nSeen:** For use in online learning, this is a tag that notes how many times the UR has been sampled.  For now, it's just for use in analyzing the learning process after the fact - it's not used during learning

**probableCandidates:** a list of all the candidates with observed probabilities greater than 5%, or 0.05

`LexEntry` also has the following methods:

`addCandidate(cand)` adds a candidate object to the list of candidates

`getProbableCandidates()` fills *probableCandidates* with the surface forms of all candidates whith observed probabilities greater than 5%.  This can be seen to approximate a Bayesian "Highest Density Interval" or "Credible Interval" of output candidates.

`decayPFC(t,decayRate,decayType='static')` Decays the phonological form constraints according to how much time has passed since the LexEntry was last observed.  The `decayRate` is a parameter defining how much they decay by.  If it's 0, they won't decay at all.  If `decayType` is set to  'static', the PFC weights will decay according to the equation: *new_weight = old_weight - (t-`lastSeen`) X decayRate*, and if the weight decays to or below zero, the lexical constraint will get removed. The `decayType` parameter defaults to `static`, yielding the above behavior.  Other options are `linear`, meaning the amount of decay depends linearly on the weight of the lexical constraint.  (less decay for lower weights), and `nonlinear`, meaning the amount of decay depends linearly on the square of the weight of the lexical constraint.

`checkViolationLength()` checks if all the violation vectors of the candidates are the same length.

`calculateHarmony(w,t=None,decayRate=None,decayType=None,suppressLexC=False)` calculates the harmony of each candidate, using w, the vector of weights of the markedness constraints.  If `decayRate` is set to something besides None, it begins by calling `decayPFC`, which is why it takes `t`, `decayRate`, and `decayType` as parameters.  If `suppressPFC` is set to True, this will calculate harmony without the phonological form constraints.  This can be useful for comparing predictions.

`predictProbs(w,t=None,decayRate=None,decayType=None, suppressPFC=False)` calculates the predicted probabilities of each candidate according to a vector of weights `w`.  Begins by calling `calculateHarmony`, which can call `decayLexC`, hence you can give it the appropriate parameters.  If `suppressLexC` is True, probabilities will be predicted without the lexical constraints.  Note that if you do this, your UR's will have the wrong predicted probabilities until you run this again without supression.  So be careful.

`getPredWinner(theory)` Get a winner based on your theory - if `MaxEnt` (currently the only one implemented), sample to get that winner.  Returns a string that's the actual surface form winner, as well as the entire candidate object that's the winner.

`getObsWinner(theory)` Get an observed winner by sampling from the observed distribution over candidates.  If there's a sole observed winner, that one will always be returned by this function.  Returns a string that's the surface form of the observed winner, and also returns the entire candidate object.

`compareObsPred(theory,w,strategy = 'sample', t=None,decayRate=None,decayType=None)` Compares observed to predicted winner, using one of two strategies.  Either samples from both the predicted and the observed distributions, and compares those samples to see if there's an error, or if `strategy` = `HDI`, it samples from the predicted distribution, and then checks whether that sampled form is one of the most probable candidates for in the observed distribution.

If `sample`, the default, it calls `getPredWinner` and `getObsWinner`, and compares the observed to the predicted winner to see if there's an error.  The function also calls `predictProbs`, which calls `calculateHarmony` and `decayLexC`, so altogether, it calculates the current values for the lexical constraints, and it calculates harmonies and probabilities given the current set of both general and lexical constraint weights, and then it samples and sees if the observed thing matches the predicted thing.  Returns whether or not there was an error (0 if no error, 1 if error), and the candidate objects for the observed candidate and the predicted candidate.  NOTE ON HIDDEN STRUCTURE:  **hidden structure still in progress.  check back later ** 



## Tableaux

use: `new_tableaux = Tableaux(theory='MaxEnt')`

_attributes_:

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
    ..* use `perceptronUpdate()` to update all the general constraints
    ..* if any lexical constraints exist for this UR, update them according to perceptron update rule.  NOTE: a hard upper limit of 700 is imposed on lexical constraint weights.  This is because around 740 or so, exponentiating the negative to calculate the probability yields a zero, which later yields a division by zero error
    ..* if there's no lexical constraint for this UR that prefers the correct output, induce one
    ..* (if lexical constraints are being induced probabilistically, first do some calculations to decide what the right probability is)
    ..*((This sampling process is still under construction))
  - increment time `t`
  - update the trained UR's `lastSeen` and `nSeen` parameters

  Ok, now let's talk about all the parameters (Many of these are brief explanations only - see the discussion of the learn() function for more details of most of them):
  - *theory*: Only 'MaxEnt' is implemented.  Hopefully in the future other types of Noisy HG will be implemented as well.

  - *learnRate*: The learning rate, or the amount by which the general constraints get updated by the perceptron update rule.  Larger values will result in faster and more granular learning.

  - *lexCstartW*: The starting weight for induction of lexically specific constraints.

  - *lexLearnRate*: The learning rate for lexical constraints

  - *lexCSample*: `False` to induce a lexical constraint on every error, `True` to induce a lexical constraint probabilistically based on how many lexical constraints are already in the system

  - *lexCSampleSize*: parameter to help with figuring out whether to induce a lexical constraint in this case

  - *decayRate*: decay rate of lexical constraints

  - *decayType*: `static`, `linear`, or `nonlinear` - see notes on `decayLexC()` to see what they all do.

  - *haveLexC*: If `True`, lexical constraints are used, if `False`, no lexical constraints.

  - *comparisonStrategy*: `sample` or `HDI` - how do you want to compare the predicted and observed outputs?  if `sample`, compare by sampling from both distributions.  If `HDI`, check whether the predicted output (sampled) has an appreciable probability (over some threshold - by default it's 0.05) in the observed distribution.

  - *urToUse*: What UR are we learning on right now?  Takes a UR object that's within the Tableaux object.

`epoch(theory, iterations, learnRate, lexCstartW, lexLearnRate, lexCSample=True, lexCSampleSize=10, decayRate=None, decayType=None, haveLexC=False, comparisonStrategy='sample')` Runs `update()` *iterations* number of times, and returns: 

- the actual error rate during the epoch - a percentage of updates which were errors
- the current sum squared error (SSE) between the observed probabilities and current predicted probabilities for each candidate.  This is calculated by the `SSE()` function.
- list of lexical constraints, in tuples of the form (ur, preferred output)
- list of weights corresponding to the lexical constraints
- current probability of inducing a lexical constraint, calculated analytically as 1-((1-NlexC/Nurs)^lexCSampleSize), which is the probability that a sample will contain at least one lexical constraint, given the sample size and percentage of URs which currently have lexical constraints affiliated with them

The reason the learning iterations are encased in these epochs is basically to limit the amount of saving of information.  So basically, the learning process is 'sampled' at the end of every epoch.  This is the primary job of the epoch function.  Here's what happens:  (1) URs are sampled, and a playlist of them is made.  (2) *iterations* updates are executed, and meanwhile information about how many of them were errors is recorded (3) catch-up decay is executed, so that all the lexical constraints are decayed the proper amount for time `t` (4) information about the epoch is structured properly to be saved and returned, so that it can be passed back up to the `learn()` function.

The parameters of `epoch()` are the same, and mean the same things as the parameters of `update()`, except for urToUse, which is specific to `update()`, and `iterations` which is specific to `epoch()` and is the number of learning updates in an epoch.

`learn(iterations, nEpochs, learnRate, lexCstartW=0, lexLearnRate=0, lexCSample=False, lexCSampleSize=10, decayRate=None, decayType='static', theory='MaxEnt', haveLexC=False, reset=True, comparisonStrategy='sample')` 

This is the main wrapper for learning, and is the function you will most likely be actually executing by hand.  It returns a `Results` object that contains samples of the learning process, and information about its end state.

- `iterations` number of learning updates in an epoch
- `nEpochs` number of epochs

The total number of learning iterations will be `iterations` x `nEpochs`. The results object will contain information about each epoch, but information about each individual iteration will not be saved.  So, you should break down your desired number of learning updates into epochs based on how finely you want to sample the learning process.  Sampling more coarsely will take less time to run (by a larger factor for larger datasets).

- `learnRate` learning rate for perceptron update of general constraints.  Use larger values for quicker, more coarse-grained learning, and smaller values for slower, more fine-grained learning.  If you have no idea what kind of number to try, start with 0.01.

`learnRate` can also be a list of two numbers, which will act as a learning rate 'schedule', so that over the course of learning the learning rate will change from the first to the second number.  The learning rate will be updated on each epoch.  For example, if `learnRate` was given as [0.1,0.01], and there were 10 epochs, the learning rate for the first epoch would be 0.1, for the second epoch would be 0.09, for the third would be 0.08 ... and for the last would be 0.01.

- `lexCstartW` Starting weight for lexical constraints when they are first induced.  Appropriate values for this depend on the decay rate and learning rate of those constraints.  You want it to be high enough that the constraints have a chance to be used before they decay away.

- `lexLearnRate` Learning rate for lexical constraints.  This is distinct from the overall learning rate to reflect the intuition that updating specific knowledge about a lexical item should be a different process than updating general grammatical knowledge.  You can also set it to be the same as `learnRate`.

- `lexCSample` If `False`, lexical constraints will be induced on every learning update which produces an error.  If `True`, determine whether a lexical constraint will be induced on error by sampling from the lexicon (URs that have previously been trained on at least once) and checking whether any of the lexical items have a lexical constraint affiliated with them.  If any lexical item in the sample does have a lexical constraint, go ahead and induce one this time.  Otherwise, don't.

Some notes on this process: At the beginning of learning, a lexical constraint is induced on every error.  Once the lexicon gets as big as the sample size, `lexCSampleSize`, start grabbing a pool of lexical items at each update.  The pool is `lexCSampleSize` number of items.  If any lexical item has a lexical constraint, it's enough.  Decide to induce a lexically specific constraint.

- `lexCSampleSize` see above.

- `decayRate` Rate of decay of the lexical constraints.  This should be pretty low to get sensible behavior (by which I mean general constraint weights and SSE stabilize over time, instead of oscillating)

- `decayType` How decay is calculated: `static` means the new weight is calculated according to the equation *new_weight = old_weight - (t-`lastSeen`) X decayRate*.  Other options are `linear`, meaning the amount of decay depends linearly on the weight of the lexical constraint.  (less decay for lower weights), and `nonlinear`, meaning the amount of decay depends linearly on the square of the weight of the lexical constraint.

- `theory` Just set it to 'MaxEnt', which is the default anyway.

- `haveLexC` If `True`, use lexical constraints.  If `False`, learn without them.

- `reset` If `True`, reset the time, weights, and lexical constraints before learning (so, learn from a fresh tableaux).  If `False`, learning just continues from where it left off.

- `comparisonStrategy` How will observed outputs be compared to predicted ones?  If `sample`, sample from both distributions and compare the samples.  If `HDI`, instead sample from the predicted distribution, and check whether the sample is in the set of 'credible outputs' in the observed distribution.  By default, this set of credible outputs is all outputs with an observed probability over 0.05.

`calcLikelihood()` Calculate log likelihood for the entire data set *NOT YET IMPLEMENTED*

`SSE()` Calculates and returns the current SSE, comparing predicted probabilities to observed probabilities for each UR and candidate.



## Society
use: `new_society = Society(generations,startTableaux,outputName)`

_attributes_:

**generations:** starts at 0.  Currently not used for anything, apparently?

**startTableaux:** A `Tableaux` object to use as the starting point for some generational learning.

**currentTableaux:** The current generation's `Tableaux` object, created from the results of learning on the last `Tableaux` object.

**resultses:** List of `Results` objects, one from each generation.

**outputName:** Prefix name to give the files that will be saved using `Results.save()` at each generation.

_methods_:

`iterate(nGenerations,iterations,nEpochs,learnRate,lexCstartW=0,lexLearnRate=0,lexCSample=False,lexCSampleSize=10,decayRate=None,decayType='static',theory='MaxEnt',haveLexC=False,reset=True,comparisonStrategy='sample',updateURprobs=True)` This function will execute `Tableaux.learn()` iteratively, using the results of each learning as the input data for the next learning run.  Most of these parameters pertain to how the learning should proceed, and you should look at `Tableaux.learn()` for help with them.  `nGenerations` states how many generations to run, and `updateURprobs` dictates whether or not the frequecies of each UR should be updated from generation to generation.  If this is False, UR probabilities will never change.  If it's True, then some low-frequency words will drop out over the course of learning, and the distribution will get more and more extremely zipfian.

`updateTableaux(candidateProbs=True,URprobs=True)` This function updates the `Society.currentTableaux` based on the results of learning.

## Results
use: This is an object that holds the results of a learning run.  You probably shouldn't just make one of these on your own.

_attributes_:

**t:** learning time.  This is going to work out to be just a list from 0 to the total number of epochs that the learner executed.  It's used basically as an index for other information, and in plots.

**w:** weight vector for the general constraints at the end of learning

**sse:** list of SSE values for the system over time, one sample for each learning epoch

**err:** list of actual error rates for each epoch (percent of iterations on that epoch that resulted in an error)

**runtime:** actual runtime of your simulation (in seconds)

**predruntime:** predicted run time - the thing that got printed at you after epoch 0.  You can use this to check the accuracy of the predictions for future references (they're probably not suuuper accurate, but hopefully within an order of magnitude)

**functionCall:** A dictionary that holds all the parameter settings that you gave the learn() function when you ran it.

**Cnames:** List of constraint names

**lexCinfo:** Ok.  This is a thing that holds the lexical constraints and their weights over time.  Its structure is as follows: It's a list of two lists [[0][1]].  List 0 is a list of tuples, each of which defines a lexical constraint (input, preferred output).  List 1 is a list of lists, each of which pertains to the corresponding lexical constraint in list 0.  Each of _these_ has two lists in it.  The first one is a list of epoch numbers, and the second is a list of weights - the weights that that lexical constraint had at each of the listed epochs.  Here's an example for you, with two constraints:

`[[(big+comp, bigger), (true+comp, more true)],[[[3,4,5][9.32,6.3,2.1]],[[8,9,10],[9.3,7.2,8.5]]]]`

The constraint (big+comp, bigger) had a weight of 9.32 at epoch 3, a weight of 6.3 at epoch 4, and a weight of 2.1 at epoch 5.  The constraint (true+comp, more true) had a weight of 9.3 at epoch 8, 7.2 at epoch 9, and 8.5 at epoch 10.


**pLexCs:** analytically calculated probability of inducing a lexically specific constraint at every epoch.  This will be meaningless unless you're using probabilistic induction

**lexicon:** The lexicon from the Tableaux object, containing a list of UR objects, and a second list of number of times each UR was used during learning.


_methods_:

`save(filename,folder)` Save your results to some text files.  This function will write, to the folder you specify, or the folder that the program is located in if you don't specify a folder, the following output files:

- metadata: this will contain a bunch of, well, metadata about your learning simulation.  It starts with the function call, then has the actual and predicted runtimes, and the final probability of inducting a lexical constraint.

- weights: The weights of your general constraints, over the course of learning.  This will be a file with columns, where the first row is the column headers.  The first column is labelled 't', and contains epoch numbers.  The remaining columns are labelled with names of the general constraints, and contain those constraints' weights for each epoch of learning.

- SSE: Two columns, one with 't', the epoch numbers, and the other with 'SSE', the sum squared error values for each epoch.

- err: Same as SSE, but with actual error rates for each epoch.

- lexicon: Two columns, 'word', containing UR's that were learned on, and 'count' indicating how many times that UR was used in learning.

- PFCinfo: Each row is the name of a lexical constraint followed by that constraint's weight at each time point.  Zeros indicate that that constraint did not exist at that time point.

- finalState: Final state of all the lexical items.  Columns: 'UR','Candidate', 'ObservedProb' (the observed probability of each candidate, the training data), 'PredictedProb' (the probability predicted by the final weights of the general constraints and all the lexical constraints), 'PredictedProbNoLexC' (the probability predicted for each candidate given the final general constraint weights, but without the lexical constraints), 'observedFreq', (actual frequency from input file), 'trainingFreq' (frequency with which that UR was used during training), 'lexCW' (weight of any lexical constraint preferring that candidate)

`plotSSE()` Create a plot of the SSE over time from learning

`plotW()` Create a plot of each constraint weight over time

`plotMeanLexW()` Make two plots: one of the number of lexical constraints present in each epoch (binned according to candidates, assuming each input has candidates that all look the same, for example 'initial stress' and 'final stress'.  This will be weird if your candidates aren't structured that way).  The second plot is the mean weight of those lexical constraints, by bin.  Note that these weights wind up being pretty zipfian, so means might not be the best way to get a handle on the behavior of each bin.


## Unaffiliated functions:

`readOTSoft(file)` Right now, this just has the capability to read in an hgr-style file and turn it into a Tableaux object, and to yell at you if the file type is wrong.

`perceptronUpdate(error, target, weights, rate)` error and target should be violation vectors, weights should be a vector of weights, and rate is the learning rate.  Returns an updated weight vector.


## To-Do list:
* Add ability to make predicted probabilities based on sampling, for results/ final state

* Add to `initializeWeights()`:
  * Ability to choose random weights, with user-specified parameters

* Figure out how to implement regularization in online updates

* Implement sensible command-line running

* Add more numpy for faster running

* Add HG, (Noisy HG? - See Bruce's AMP talk for some sensible varieties)


## Wish list:

* Implement option for different learning rates for each general constraint, a la Jesney and Tessier

* Implement batch gradient descent option
  * Add lexC generation
  * Figure out what to do with hidden structure

* Add stress candidate generation
