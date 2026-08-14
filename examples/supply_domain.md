# Marketplace operations — how the business works

*An internal briefing for people joining the operations team. Describes the business, not any
system that models it.*

## What we are

We run an online marketplace in Brazil. We do not own inventory. Independent sellers list what
they have, shoppers buy it, and we sit in the middle: we take the payment, we set the delivery
promise, and we are the name the shopper blames when something goes wrong. That last part is why
this briefing exists.

## How a purchase works

A shopper puts things in a basket and pays once. What arrives may come in several parcels on
several days, because the things in one basket often come from different sellers in different
states. Each line of a purchase is dispatched on its own: it has its own seller, its own parcel,
its own journey. A shopper thinks of the purchase; operations has to think of the lines.

The same item is frequently offered by more than one seller, at different prices and from
different places. Which seller a shopper ends up buying from is decided at checkout, so the
seller is a fact about the line, not about the item.

## The delivery promise

At checkout we show the shopper a date. That date is ours, not the seller's — we calculate it
from the origin and destination and add a margin. Everything downstream hangs off it. A parcel
that arrives after the shown date is late, however fast it actually travelled, and one that
arrives before it is on time even if it sat in a depot for a week.

Two consequences of this are worth understanding early.

First, when we judge sellers on punctuality we are judging them against a promise we made on
their behalf. A seller in a remote state gets a generous date and looks excellent; a seller next
door gets a tight one and looks poor. The number is real, and it is not a clean measure of
performance.

Second, a shopper who is not told about a delay finds out on the day. Under Brazilian consumer
law we are required to notify the shopper when a purchase will materially miss its date, and
"materially" has been interpreted internally as three days or more. This is an obligation, not
a courtesy. If the delay is one day we may notify and often do; if it is three, staying quiet is
a breach, and the breach is ours, not the seller's.

## What sellers can and cannot do

A seller ships. That is the ordinary case and needs no discussion.

Some items cannot go on the standard service. Couriers price and route by weight and by size,
and above a certain weight the standard contract simply does not cover the parcel — it needs the
freight service, which is slower and costs us more. Where that line sits is a commercial decision
we renegotiate with couriers, so it moves. Anything over it is what the team calls a bulky item.
A bulky item is still an ordinary item in every other respect: it has a price, a category, a
seller, it appears in purchases like anything else. It is one specific thing about it — how it
must be shipped — that differs.

## When a seller is failing

We track, for each seller, the share of their delivered purchases that arrived by the promised
date. Below 80% we consider suspending them from the marketplace pending review.

The judgement is harder than the number suggests, and this is the part new joiners get wrong.
Most of our sellers are small. A seller who has delivered three purchases and was late on one is
at 67%, far below the threshold, on evidence that would flip entirely with one more delivery. We
do not suspend on that. A rate computed over a handful of purchases is noise, and suspension
takes away someone's income. The rule the team follows is that we do not act on fewer than twenty
delivered purchases, regardless of what the percentage says. That is a rule about what we are
permitted to do, not about what the number means — the number is what it is, and we still may not
act on it.

Suspension is also not something an operations analyst does alone. It is an institutional act:
it changes the seller's standing on the platform, and it is reviewed before it takes effect.

## What we do not control

Couriers strike. Regional floods close roads. A carrier stops accepting parcels for a whole
state, sometimes for a week. None of this is anybody's decision here, and none of it can be
prevented by policy. It still has to be understood, because it is usually the reason a week's
punctuality figures collapse, and because a seller's rate should not be read as a judgement of
that seller during one of these events.

## Who is who

Shoppers are individuals, and we hold little about them beyond where they are — the state
matters because it drives the delivery estimate and the freight cost. Sellers are businesses,
also identified by state, and unlike shoppers they have a history with us that we score.
