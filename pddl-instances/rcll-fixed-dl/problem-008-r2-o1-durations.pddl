(define (problem rcll-production-008-durative)
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
		(order-gate o1 GATE-3)
		(= (path-length C-BS INPUT C-BS OUTPUT) 2.850827)
		(= (path-length C-BS INPUT C-CS1 INPUT) 9.902102)
		(= (path-length C-BS INPUT C-CS1 OUTPUT) 7.140012)
		(= (path-length C-BS INPUT C-CS2 INPUT) 2.919863)
		(= (path-length C-BS INPUT C-CS2 OUTPUT) 6.61816)
		(= (path-length C-BS INPUT C-DS INPUT) 7.972726)
		(= (path-length C-BS INPUT C-DS OUTPUT) 6.608266)
		(= (path-length C-BS OUTPUT C-BS INPUT) 2.850827)
		(= (path-length C-BS OUTPUT C-CS1 INPUT) 8.689058)
		(= (path-length C-BS OUTPUT C-CS1 OUTPUT) 5.926968)
		(= (path-length C-BS OUTPUT C-CS2 INPUT) 3.443113)
		(= (path-length C-BS OUTPUT C-CS2 OUTPUT) 5.43461)
		(= (path-length C-BS OUTPUT C-DS INPUT) 6.759683)
		(= (path-length C-BS OUTPUT C-DS OUTPUT) 6.480538)
		(= (path-length C-CS1 INPUT C-BS INPUT) 9.902102)
		(= (path-length C-CS1 INPUT C-BS OUTPUT) 8.689058)
		(= (path-length C-CS1 INPUT C-CS1 OUTPUT) 4.901756)
		(= (path-length C-CS1 INPUT C-CS2 INPUT) 8.928803)
		(= (path-length C-CS1 INPUT C-CS2 OUTPUT) 6.256728)
		(= (path-length C-CS1 INPUT C-DS INPUT) 4.48762)
		(= (path-length C-CS1 INPUT C-DS OUTPUT) 7.024663)
		(= (path-length C-CS1 OUTPUT C-BS INPUT) 7.140013)
		(= (path-length C-CS1 OUTPUT C-BS OUTPUT) 5.926968)
		(= (path-length C-CS1 OUTPUT C-CS1 INPUT) 4.901756)
		(= (path-length C-CS1 OUTPUT C-CS2 INPUT) 6.166713)
		(= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 4.702538)
		(= (path-length C-CS1 OUTPUT C-DS INPUT) 2.972381)
		(= (path-length C-CS1 OUTPUT C-DS OUTPUT) 5.509424)
		(= (path-length C-CS2 INPUT C-BS INPUT) 2.919863)
		(= (path-length C-CS2 INPUT C-BS OUTPUT) 3.443113)
		(= (path-length C-CS2 INPUT C-CS1 INPUT) 8.928802)
		(= (path-length C-CS2 INPUT C-CS1 OUTPUT) 6.166713)
		(= (path-length C-CS2 INPUT C-CS2 OUTPUT) 4.186741)
		(= (path-length C-CS2 INPUT C-DS INPUT) 6.860031)
		(= (path-length C-CS2 INPUT C-DS OUTPUT) 4.176848)
		(= (path-length C-CS2 OUTPUT C-BS INPUT) 6.61816)
		(= (path-length C-CS2 OUTPUT C-BS OUTPUT) 5.43461)
		(= (path-length C-CS2 OUTPUT C-CS1 INPUT) 6.256728)
		(= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 4.702538)
		(= (path-length C-CS2 OUTPUT C-CS2 INPUT) 4.186741)
		(= (path-length C-CS2 OUTPUT C-DS INPUT) 4.327353)
		(= (path-length C-CS2 OUTPUT C-DS OUTPUT) 1.918192)
		(= (path-length C-DS INPUT C-BS INPUT) 7.972727)
		(= (path-length C-DS INPUT C-BS OUTPUT) 6.759683)
		(= (path-length C-DS INPUT C-CS1 INPUT) 4.48762)
		(= (path-length C-DS INPUT C-CS1 OUTPUT) 2.972381)
		(= (path-length C-DS INPUT C-CS2 INPUT) 6.860031)
		(= (path-length C-DS INPUT C-CS2 OUTPUT) 4.327353)
		(= (path-length C-DS INPUT C-DS OUTPUT) 3.083661)
		(= (path-length C-DS OUTPUT C-BS INPUT) 6.608267)
		(= (path-length C-DS OUTPUT C-BS OUTPUT) 6.480538)
		(= (path-length C-DS OUTPUT C-CS1 INPUT) 7.024663)
		(= (path-length C-DS OUTPUT C-CS1 OUTPUT) 5.509423)
		(= (path-length C-DS OUTPUT C-CS2 INPUT) 4.176848)
		(= (path-length C-DS OUTPUT C-CS2 OUTPUT) 1.918192)
		(= (path-length C-DS OUTPUT C-DS INPUT) 3.08366)
		(= (path-length START INPUT C-BS INPUT) 3.737139)
		(= (path-length START INPUT C-BS OUTPUT) 1.461364)
		(= (path-length START INPUT C-CS1 INPUT) 8.150055)
		(= (path-length START INPUT C-CS1 OUTPUT) 5.387965)
		(= (path-length START INPUT C-CS2 INPUT) 3.274815)
		(= (path-length START INPUT C-CS2 OUTPUT) 4.895607)
		(= (path-length START INPUT C-DS INPUT) 6.22068)
		(= (path-length START INPUT C-DS OUTPUT) 5.941535)
	)
	(:goal (order-fulfilled o1))
)