(define (problem rcll-production-033-durative)
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
		(order-base-color o1 BASE_BLACK)
		(order-cap-color o1 CAP_GREY)
		(order-gate o1 GATE-3)
		(= (path-length C-BS INPUT C-BS OUTPUT) 2.656903)
		(= (path-length C-BS INPUT C-CS1 INPUT) 5.330379)
		(= (path-length C-BS INPUT C-CS1 OUTPUT) 6.739193)
		(= (path-length C-BS INPUT C-CS2 INPUT) 10.795329)
		(= (path-length C-BS INPUT C-CS2 OUTPUT) 10.905421)
		(= (path-length C-BS INPUT C-DS INPUT) 7.276423)
		(= (path-length C-BS INPUT C-DS OUTPUT) 6.38295)
		(= (path-length C-BS OUTPUT C-BS INPUT) 2.656903)
		(= (path-length C-BS OUTPUT C-CS1 INPUT) 3.544793)
		(= (path-length C-BS OUTPUT C-CS1 OUTPUT) 5.579175)
		(= (path-length C-BS OUTPUT C-CS2 INPUT) 10.917266)
		(= (path-length C-BS OUTPUT C-CS2 OUTPUT) 11.027359)
		(= (path-length C-BS OUTPUT C-DS INPUT) 7.398361)
		(= (path-length C-BS OUTPUT C-DS OUTPUT) 6.504888)
		(= (path-length C-CS1 INPUT C-BS INPUT) 5.330379)
		(= (path-length C-CS1 INPUT C-BS OUTPUT) 3.544793)
		(= (path-length C-CS1 INPUT C-CS1 OUTPUT) 3.098388)
		(= (path-length C-CS1 INPUT C-CS2 INPUT) 11.381385)
		(= (path-length C-CS1 INPUT C-CS2 OUTPUT) 10.467551)
		(= (path-length C-CS1 INPUT C-DS INPUT) 7.862479)
		(= (path-length C-CS1 INPUT C-DS OUTPUT) 5.653862)
		(= (path-length C-CS1 OUTPUT C-BS INPUT) 6.739194)
		(= (path-length C-CS1 OUTPUT C-BS OUTPUT) 5.579175)
		(= (path-length C-CS1 OUTPUT C-CS1 INPUT) 3.098388)
		(= (path-length C-CS1 OUTPUT C-CS2 INPUT) 10.498308)
		(= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 7.620886)
		(= (path-length C-CS1 OUTPUT C-DS INPUT) 5.699126)
		(= (path-length C-CS1 OUTPUT C-DS OUTPUT) 2.807196)
		(= (path-length C-CS2 INPUT C-BS INPUT) 10.795328)
		(= (path-length C-CS2 INPUT C-BS OUTPUT) 10.917266)
		(= (path-length C-CS2 INPUT C-CS1 INPUT) 11.381386)
		(= (path-length C-CS2 INPUT C-CS1 OUTPUT) 10.498308)
		(= (path-length C-CS2 INPUT C-CS2 OUTPUT) 2.980464)
		(= (path-length C-CS2 INPUT C-DS INPUT) 7.025482)
		(= (path-length C-CS2 INPUT C-DS OUTPUT) 7.898801)
		(= (path-length C-CS2 OUTPUT C-BS INPUT) 10.905421)
		(= (path-length C-CS2 OUTPUT C-BS OUTPUT) 11.027359)
		(= (path-length C-CS2 OUTPUT C-CS1 INPUT) 10.467551)
		(= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 7.620886)
		(= (path-length C-CS2 OUTPUT C-CS2 INPUT) 2.980464)
		(= (path-length C-CS2 OUTPUT C-DS INPUT) 4.455125)
		(= (path-length C-CS2 OUTPUT C-DS OUTPUT) 5.021379)
		(= (path-length C-DS INPUT C-BS INPUT) 7.276423)
		(= (path-length C-DS INPUT C-BS OUTPUT) 7.398361)
		(= (path-length C-DS INPUT C-CS1 INPUT) 7.86248)
		(= (path-length C-DS INPUT C-CS1 OUTPUT) 5.699126)
		(= (path-length C-DS INPUT C-CS2 INPUT) 7.025483)
		(= (path-length C-DS INPUT C-CS2 OUTPUT) 4.455125)
		(= (path-length C-DS INPUT C-DS OUTPUT) 3.09962)
		(= (path-length C-DS OUTPUT C-BS INPUT) 6.38295)
		(= (path-length C-DS OUTPUT C-BS OUTPUT) 6.504889)
		(= (path-length C-DS OUTPUT C-CS1 INPUT) 5.653862)
		(= (path-length C-DS OUTPUT C-CS1 OUTPUT) 2.807196)
		(= (path-length C-DS OUTPUT C-CS2 INPUT) 7.898802)
		(= (path-length C-DS OUTPUT C-CS2 OUTPUT) 5.021379)
		(= (path-length C-DS OUTPUT C-DS INPUT) 3.099619)
		(= (path-length START INPUT C-BS INPUT) 1.577976)
		(= (path-length START INPUT C-BS OUTPUT) 3.1291)
		(= (path-length START INPUT C-CS1 INPUT) 5.962154)
		(= (path-length START INPUT C-CS1 OUTPUT) 5.922573)
		(= (path-length START INPUT C-CS2 INPUT) 9.978709)
		(= (path-length START INPUT C-CS2 OUTPUT) 10.088801)
		(= (path-length START INPUT C-DS INPUT) 6.459803)
		(= (path-length START INPUT C-DS OUTPUT) 5.56633)
	)
	(:goal (order-fulfilled o1))
)