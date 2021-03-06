# Logistics-datascience-challenge
Solution to a Data Science challenge aimed at optimising logistics choices in the Oil&Gas industry  
A description of the problem follows below; for a more detailed version, refer to *challenge_statement.pdf*

1. CONTEXT

Imagine an Oil &amp; Gas operator with offshore assets (e.g. platforms). These assets require regular visits from staff to carry out safety inspections, maintenance activities, interventions, etc. The _planning_ of these visits is often complex due to different _value drivers_ (e.g., not all activities are equally important and/or urgent) and _uncertainties_ (e.g., weather may change on a daily basis and/or transport vehicles may be temporarily out of service).

2. BUSINESS QUESTION

How to _optimally allocate staff to transport vehicles_ (e.g. helicopters), if the aim is to meet demand with maximum vehicle efficiency whilst being robust against uncertainty?

1. GIVEN

Assume the following:

3.1 SUPPLY OF STAFF

- The total number of available staff varies from day to day. Assume the total staff count can be described with a gaussian distribution with mean 60 and standard deviation 20.
- Staff are organized in teams with team sizes varying between 1 and 8. Teams cannot be split.

3.2 SUPPLY OF VEHICLES

- The operator has 4 helicopters. Two of them can carry 25 staff each, the others can carry 15 staff each.
- Helicopters can stop at more than 1 location within a single trip.
- Flight time is assumed negligible.

3.3 DEMAND

- The operator has 10 offshore locations that need maintaining.
- Demand across locations varies uniformly on a daily basis. That is, each day demand is randomly distributed across all locations.

- Each team is assigned to one offshore location. Multiple teams may be assigned to the same location, but a single team is never assigned to more than one location.

3.4 OPTIMIZATION

- The operator&#39;s aim is to allocate as much staff as possible with as few helicopters as possible.

4. DELIVERABLE

Design one or more approaches that are capable of addressing the business question. Demonstrate the efficiency &amp; robustness of your approach(es).


![staff allocation example](https://github.com/achiappo/Logistics-datascience-challenge/blob/main/logistics.png?raw=true)
