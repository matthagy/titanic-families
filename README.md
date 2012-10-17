Determine the families in the Kaggle Titanic competition
------------------------------------------------------------
Heuristic-based algorithm to find the family trees in the titanic data.

Only involves a few parameters, all of which are related to age
(e.g. minimum age for marriage). See the constant defined at the
top for all of them. Neither iterative nor stochastic methods are
used.

First a graph of individuals is constructed where the edges represent
shared last names. This includes any previous names such maiden names.
Each edge represents as relationship that can be classified as one
of the following:
   * Spouse
   * Parent/Child
   * Sibling
   * Extended (e.g. aunt, cousin, or distant relative)

The classification scheme is optimistic, i.e. we only ask
whether or not the relationship is possible. Much of information
can be directly inferred from the given attributes (e.g. two
individual cannot be siblings if one of them has sibsp==0).

Next we prove spousal relationship. This fairly easy, epically
as many spouses name pairs are of the form:

    West, Mrs. Edwy Arthur (Ada Mary Worth)
    West, Mr. Edwy Arthur

We don't require names of this style and can also use age differences,
requires Mrs title for the female, and other simple heuristics.
The only difficulty arises when one individual could be classified as
married to multiple individuals. There are only a few such situations and
they can all be handled by assigning marriage to the couple in which the
female has the males first name (e.g. Mrs. Edwy Arthur).

With spousal relationships found it is then straightforward to workout
parent/child relationships. The only ambiguities at this point are
child vs. sibling and they can be resolved by checking for common
parent(s). Lastly, parent/child relationships can be used to work out
sibling relationships. We can then recover the structure of nuclear
families: families in which there is at least one parent and one or more
children.

Outside of the nuclear family structure, we still maintain the
relationship graph which allows for such classifications as:
  * siblings traveling together without any parents
  * extended relations
  * families joined by extended relationships

At the moment there are still some edge cases. In particular, the largest
relationship graph component isn't separated into a family structure.
Additionally, it would be nice to remove or relax the few parameters.

Project is licensed under the BSD (2-clause) license
(see include LICENSE file)
