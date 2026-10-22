# Prosperity




## Useful Files

- `traders/round1/ash_coated_osmium_only.py`
  Current standalone bot for `ASH_COATED_OSMIUM`

UNDER ANALYSIS THE IMAGES BIT

- `backtesting/fresh_backtester.py`
  Local backtester

The Analysis bit is a bot that worked for the last commodity, it doesn't really transfer over for the traits of one of the new ones. the INtrarian pepper root now.

`INTARIAN_PEPPER_ROOT` is still under investigation.

Therefore this bit 

- `traders/round1/round1_anchor_trader.py`
  Early round-1 combined anchor-style trader

Is not actually very accurate at the moment, it transfers both of the logics, so loses money on the pepper root

The other commodity transfers quite well, it was the Emerald in round0 code, now its ash coated osmium only in the round 1 code, its performing quite well,

The idea is fair value is basically held throughout the whole time, and you just trade the spread essentially, having a solid anchor to work volume around

The backtester worked for the tutorial, means you dont have to upload every iteration of your code to check the PNL for it, this backtester has tested to have pretty similar logic, so it gives a similar value each time, if you use a unique backtester sometimes algos will over perform locally and vice versa, it can probably still be improved in accuracy, will keep testing it this round to make sure it holds up.
