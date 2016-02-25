# Online learner with lexically specific constraints

## Classes:

### candidate
use: `new_candidate = candidate(c,violations,observedProb,surfaceForm=None)`

**c:** the candidate

**violations:** list of violations

**observedProb:** observed probability of this candidate, from 0-1 1 if it is the winner

**surfaceForm:** for use in hidden structure situations, this is the surface form that corresponds to the actual candidate.  For example, a stress candidate might be (10)0, and its surface form 100.  The surface form for (1)00, which has a different violation profile, would also be 100.

`candidate` also has the following attributes:

**harmony:** The candidate's harmony

**predictedProb:** Candidate's predicted probability (to be compared at some point with `observedProb`)

And, it has the following methods:

`checkViolationsSign()` runs automatically on initialization of a new instance of the class, and converts violation profiles to negative floats.  It also converts blanks to 0's, but it will throw an error if any violations are made of text that can't be converted to floats.

### UR
use: `new_UR = UR(ur,prob=1,lexC=None)`

**ur:** a string, the UR, or the input if you prefer (Note: it's called UR because 'input' is a function in python, not because of any theoretical commitments.)

**prob:** you can use this attribute to define a token frequency for this UR, for example if you want to sample or weight URs/inputs by token frequency

**lexC:** This starts out empty, and can stay empty if you're not using lexically-specific constraints.  It should typically not be defined in advance, but you can if you need to.  It should wind up being a list of tuples, of the form `(output, weight)` where `output` is a string corresponding to the output that the lexically-specific constraint demands, and `weight` is the weight on that constraint.  __Lexical constraints are hitched to URs/inputs, and are not stored with the other constraints.  This might need to be changed to allow for batch learning __

`UR` also has the following attributes:

**candidates:** A list containing objects of the class `candidate`

**probDenom:** This is for calculating MaxEnt probability for candidates of this input.  It's just the denominator for that calculation: sum(e^(-H))

**predProbsList:** List of each candidate's predicted probability, for use in sampling

**lastSeen:** For use in online learning, this is a tag that notes when the UR was last sampled.  It's used for decaying the lexically-specific constraints

`UR` also has the following methods:

`addCandidate(cand)` adds a candidate to the list of candidates

`decayLexC(t,decayRate,decayType='linear')` Decays the lexical constraints according to how much time has passed since the UR was observed.  The `decayRate` is a parameter defining how much they decay by.  If it's 0, they won't decay at all.  `decayType` can be `linear` or `logarithmic`.  If `linear`, the lexical constraint weights will decay according to the equation: *new_weight = old_weight - (t-`lastSeen`) X decayRate*, and if the weight decays to or below zero, the lexical constraint will get removed. If decay is set to `logarithmic`, the weights will decay according to *new_weight = old_weight - (t-lastSeen)^decayRate*.  Once again, if it gets to or below zero the constraint will get removed.

`checkViolationLength()` checks if all the violation vectors of the candidates are the same length ** Not written yet**

`calculateHarmony(w,t=None,decayRate=None,decayType=None)` calculates the harmony of each candidate, using w, the vector of weights of the markedness constraints.  If `decayRate` is set to something besides None, it begins by calling `decayLexC`, which is why it takes `t`, `decayRate`, and `decayType` as parameters.

`predictProbs(w,t=None,decayRate=None,decayType=None)` calculates the predicted probabilities of each candidate according to a vector of weights `w`.  Begins by calling `calculateHarmony`, which can call `decayLexC`, hence you can give it the appropriate parameters

`getWinner(theory)` Get a winner.  **In progress**

## Tableaux

use: `new_tableaux = Tableaux(theory='MaxEnt')`

attributes:

**urList:** list of objects of class `UR`

**urIDlist:** list of strings, that are the strings of the URs

**theory:** Right now, only 'MaxEnt' is implemented

methods:

`addUR(ur)` adds an object of class `UR` to the `Tableaux`

## Functions:

`readOTSoft(file)` Right now, this just has the capability to read in an hgr-style file and turn it into a Tableaux object, and to yell at you if the file type is wrong.
