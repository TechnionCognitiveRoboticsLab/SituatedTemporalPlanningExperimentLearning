(define (problem rcll-production-018-durative)
	(:domain rcll-production-durative)
	(:objects R-1 - robot R-2 - robot o1 - order wp1 - workpiece cg1 - cap-carrier cg2 - cap-carrier cg3 - cap-carrier cb1 - cap-carrier cb2 - cap-carrier cb3 - cap-carrier C-BS - mps C-CS1 - mps C-CS2 - mps C-DS - mps CYAN - team-color)
	(:init 
		(order-delivery-window-open o1)
		(at 150 (not (order-delivery-window-open o1)))
		(can-commit-for-ontime-delivery o1)
		(at 15 (not (can-commit-for-ontime-delivery o1)))
		(mps-type C-BS BS)
		(mps-type C-CS1 CS)
		(mps-type C-CS2 CS)
		(mps-type C-DS DS)
		(location-free START INPUT)
		(location-free C-BS INPUT)
		(location-free C-BS OUTPUT)
		(location-free C-CS1 INPUT)
		(location-free C-CS1 OUTPUT)
		(location-free C-CS2 INPUT)
		(location-free C-CS2 OUTPUT)
		(location-free C-DS INPUT)
		(location-free C-DS OUTPUT)
		(cs-can-perform C-CS1 CS_RETRIEVE)
		(cs-can-perform C-CS2 CS_RETRIEVE)
		(cs-free C-CS1)
		(cs-free C-CS2)
		(wp-base-color wp1 BASE_NONE)
		(wp-cap-color wp1 CAP_NONE)
		(wp-ring1-color wp1 RING_NONE)
		(wp-ring2-color wp1 RING_NONE)
		(wp-ring3-color wp1 RING_NONE)
		(wp-unused wp1)
		(robot-waiting R-1)
		(robot-waiting R-2)
		(mps-state C-BS IDLE)
		(mps-state C-CS1 IDLE)
		(mps-state C-CS2 IDLE)
		(mps-state C-DS IDLE)
		(wp-cap-color cg1 CAP_GREY)
		(wp-cap-color cg2 CAP_GREY)
		(wp-cap-color cg3 CAP_GREY)
		(wp-on-shelf cg1 C-CS1 LEFT)
		(wp-on-shelf cg2 C-CS1 MIDDLE)
		(wp-on-shelf cg3 C-CS1 RIGHT)
		(wp-cap-color cb1 CAP_BLACK)
		(wp-cap-color cb2 CAP_BLACK)
		(wp-cap-color cb3 CAP_BLACK)
		(wp-on-shelf cb1 C-CS2 LEFT)
		(wp-on-shelf cb2 C-CS2 MIDDLE)
		(wp-on-shelf cb3 C-CS2 RIGHT)
		(order-complexity o1 c0)
		(order-base-color o1 BASE_RED)
		(order-cap-color o1 CAP_GREY)
		(order-gate o1 GATE-2)
		(= (path-length C-BS INPUT C-BS OUTPUT) 2.720804)
		(= (path-length C-BS INPUT C-CS1 INPUT) 9.565592)
		(= (path-length C-BS INPUT C-CS1 OUTPUT) 8.548266)
		(= (path-length C-BS INPUT C-CS2 INPUT) 2.307705)
		(= (path-length C-BS INPUT C-CS2 OUTPUT) 3.689645)
		(= (path-length C-BS INPUT C-DS INPUT) 7.466354)
		(= (path-length C-BS INPUT C-DS OUTPUT) 7.302048)
		(= (path-length C-BS OUTPUT C-BS INPUT) 2.720804)
		(= (path-length C-BS OUTPUT C-CS1 INPUT) 10.167015)
		(= (path-length C-BS OUTPUT C-CS1 OUTPUT) 9.149689)
		(= (path-length C-BS OUTPUT C-CS2 INPUT) 0.943229)
		(= (path-length C-BS OUTPUT C-CS2 OUTPUT) 2.105561)
		(= (path-length C-BS OUTPUT C-DS INPUT) 8.067778)
		(= (path-length C-BS OUTPUT C-DS OUTPUT) 6.470572)
		(= (path-length C-CS1 INPUT C-BS INPUT) 9.565591)
		(= (path-length C-CS1 INPUT C-BS OUTPUT) 10.167014)
		(= (path-length C-CS1 INPUT C-CS1 OUTPUT) 4.050063)
		(= (path-length C-CS1 INPUT C-CS2 INPUT) 9.333467)
		(= (path-length C-CS1 INPUT C-CS2 OUTPUT) 11.46239)
		(= (path-length C-CS1 INPUT C-DS INPUT) 7.109564)
		(= (path-length C-CS1 INPUT C-DS OUTPUT) 8.077049)
		(= (path-length C-CS1 OUTPUT C-BS INPUT) 8.548265)
		(= (path-length C-CS1 OUTPUT C-BS OUTPUT) 9.149689)
		(= (path-length C-CS1 OUTPUT C-CS1 INPUT) 4.050063)
		(= (path-length C-CS1 OUTPUT C-CS2 INPUT) 8.316141)
		(= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 9.148787)
		(= (path-length C-CS1 OUTPUT C-DS INPUT) 4.795962)
		(= (path-length C-CS1 OUTPUT C-DS OUTPUT) 6.584468)
		(= (path-length C-CS2 INPUT C-BS INPUT) 2.307705)
		(= (path-length C-CS2 INPUT C-BS OUTPUT) 0.943229)
		(= (path-length C-CS2 INPUT C-CS1 INPUT) 9.333468)
		(= (path-length C-CS2 INPUT C-CS1 OUTPUT) 8.316142)
		(= (path-length C-CS2 INPUT C-CS2 OUTPUT) 2.752124)
		(= (path-length C-CS2 INPUT C-DS INPUT) 7.23423)
		(= (path-length C-CS2 INPUT C-DS OUTPUT) 5.637025)
		(= (path-length C-CS2 OUTPUT C-BS INPUT) 3.689645)
		(= (path-length C-CS2 OUTPUT C-BS OUTPUT) 2.105561)
		(= (path-length C-CS2 OUTPUT C-CS1 INPUT) 11.462391)
		(= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 9.148788)
		(= (path-length C-CS2 OUTPUT C-CS2 INPUT) 2.752124)
		(= (path-length C-CS2 OUTPUT C-DS INPUT) 8.066876)
		(= (path-length C-CS2 OUTPUT C-DS OUTPUT) 5.361102)
		(= (path-length C-DS INPUT C-BS INPUT) 7.466353)
		(= (path-length C-DS INPUT C-BS OUTPUT) 8.067777)
		(= (path-length C-DS INPUT C-CS1 INPUT) 7.109563)
		(= (path-length C-DS INPUT C-CS1 OUTPUT) 4.795962)
		(= (path-length C-DS INPUT C-CS2 INPUT) 7.234229)
		(= (path-length C-DS INPUT C-CS2 OUTPUT) 8.066876)
		(= (path-length C-DS INPUT C-DS OUTPUT) 3.188922)
		(= (path-length C-DS OUTPUT C-BS INPUT) 7.302049)
		(= (path-length C-DS OUTPUT C-BS OUTPUT) 6.470572)
		(= (path-length C-DS OUTPUT C-CS1 INPUT) 8.07705)
		(= (path-length C-DS OUTPUT C-CS1 OUTPUT) 6.584467)
		(= (path-length C-DS OUTPUT C-CS2 INPUT) 5.637025)
		(= (path-length C-DS OUTPUT C-CS2 OUTPUT) 5.361102)
		(= (path-length C-DS OUTPUT C-DS INPUT) 3.188922)
		(= (path-length START INPUT C-BS INPUT) 1.655497)
		(= (path-length START INPUT C-BS OUTPUT) 2.25692)
		(= (path-length START INPUT C-CS1 INPUT) 8.587034)
		(= (path-length START INPUT C-CS1 OUTPUT) 7.569708)
		(= (path-length START INPUT C-CS2 INPUT) 1.423372)
		(= (path-length START INPUT C-CS2 OUTPUT) 4.065814)
		(= (path-length START INPUT C-DS INPUT) 6.487796)
		(= (path-length START INPUT C-DS OUTPUT) 6.417716)
	)
	(:goal (order-fulfilled o1))
)