(define (problem rcll-production-085-durative)
	(:domain rcll-production-durative)
	(:objects R-1 - robot R-2 - robot R-3 - robot o1 - order wp1 - workpiece cg1 - cap-carrier cg2 - cap-carrier cg3 - cap-carrier cb1 - cap-carrier cb2 - cap-carrier cb3 - cap-carrier C-BS - mps C-CS1 - mps C-CS2 - mps C-DS - mps CYAN - team-color)
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
		(robot-waiting R-3)
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
		(order-base-color o1 BASE_SILVER)
		(order-cap-color o1 CAP_GREY)
		(order-gate o1 GATE-1)
		(= (path-length C-BS INPUT C-BS OUTPUT) 2.222936)
		(= (path-length C-BS INPUT C-CS1 INPUT) 8.106909)
		(= (path-length C-BS INPUT C-CS1 OUTPUT) 6.735013)
		(= (path-length C-BS INPUT C-CS2 INPUT) 9.680581)
		(= (path-length C-BS INPUT C-CS2 OUTPUT) 9.924994)
		(= (path-length C-BS INPUT C-DS INPUT) 8.572544)
		(= (path-length C-BS INPUT C-DS OUTPUT) 6.494176)
		(= (path-length C-BS OUTPUT C-BS INPUT) 2.222937)
		(= (path-length C-BS OUTPUT C-CS1 INPUT) 6.220351)
		(= (path-length C-BS OUTPUT C-CS1 OUTPUT) 5.657924)
		(= (path-length C-BS OUTPUT C-CS2 INPUT) 7.794024)
		(= (path-length C-BS OUTPUT C-CS2 OUTPUT) 8.038437)
		(= (path-length C-BS OUTPUT C-DS INPUT) 7.983819)
		(= (path-length C-BS OUTPUT C-DS OUTPUT) 5.417087)
		(= (path-length C-CS1 INPUT C-BS INPUT) 8.106909)
		(= (path-length C-CS1 INPUT C-BS OUTPUT) 6.220351)
		(= (path-length C-CS1 INPUT C-CS1 OUTPUT) 3.322777)
		(= (path-length C-CS1 INPUT C-CS2 INPUT) 4.893129)
		(= (path-length C-CS1 INPUT C-CS2 OUTPUT) 5.137543)
		(= (path-length C-CS1 INPUT C-DS INPUT) 3.342485)
		(= (path-length C-CS1 INPUT C-DS OUTPUT) 3.58527)
		(= (path-length C-CS1 OUTPUT C-BS INPUT) 6.735013)
		(= (path-length C-CS1 OUTPUT C-BS OUTPUT) 5.657924)
		(= (path-length C-CS1 OUTPUT C-CS1 INPUT) 3.322777)
		(= (path-length C-CS1 OUTPUT C-CS2 INPUT) 6.214959)
		(= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 7.243543)
		(= (path-length C-CS1 OUTPUT C-DS INPUT) 2.835236)
		(= (path-length C-CS1 OUTPUT C-DS OUTPUT) 0.45986)
		(= (path-length C-CS2 INPUT C-BS INPUT) 9.680582)
		(= (path-length C-CS2 INPUT C-BS OUTPUT) 7.794024)
		(= (path-length C-CS2 INPUT C-CS1 INPUT) 4.893129)
		(= (path-length C-CS2 INPUT C-CS1 OUTPUT) 6.21496)
		(= (path-length C-CS2 INPUT C-CS2 OUTPUT) 4.28491)
		(= (path-length C-CS2 INPUT C-DS INPUT) 6.234667)
		(= (path-length C-CS2 INPUT C-DS OUTPUT) 6.477453)
		(= (path-length C-CS2 OUTPUT C-BS INPUT) 9.924995)
		(= (path-length C-CS2 OUTPUT C-BS OUTPUT) 8.038437)
		(= (path-length C-CS2 OUTPUT C-CS1 INPUT) 5.137543)
		(= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 7.243544)
		(= (path-length C-CS2 OUTPUT C-CS2 INPUT) 4.28491)
		(= (path-length C-CS2 OUTPUT C-DS INPUT) 7.263252)
		(= (path-length C-CS2 OUTPUT C-DS OUTPUT) 7.506037)
		(= (path-length C-DS INPUT C-BS INPUT) 8.572543)
		(= (path-length C-DS INPUT C-BS OUTPUT) 7.983818)
		(= (path-length C-DS INPUT C-CS1 INPUT) 3.342485)
		(= (path-length C-DS INPUT C-CS1 OUTPUT) 2.835236)
		(= (path-length C-DS INPUT C-CS2 INPUT) 6.234667)
		(= (path-length C-DS INPUT C-CS2 OUTPUT) 7.263251)
		(= (path-length C-DS INPUT C-DS OUTPUT) 3.097728)
		(= (path-length C-DS OUTPUT C-BS INPUT) 6.494176)
		(= (path-length C-DS OUTPUT C-BS OUTPUT) 5.417087)
		(= (path-length C-DS OUTPUT C-CS1 INPUT) 3.58527)
		(= (path-length C-DS OUTPUT C-CS1 OUTPUT) 0.45986)
		(= (path-length C-DS OUTPUT C-CS2 INPUT) 6.477452)
		(= (path-length C-DS OUTPUT C-CS2 OUTPUT) 7.506036)
		(= (path-length C-DS OUTPUT C-DS INPUT) 3.097729)
		(= (path-length START INPUT C-BS INPUT) 2.673301)
		(= (path-length START INPUT C-BS OUTPUT) 0.755138)
		(= (path-length START INPUT C-CS1 INPUT) 5.527008)
		(= (path-length START INPUT C-CS1 OUTPUT) 5.733999)
		(= (path-length START INPUT C-CS2 INPUT) 7.100679)
		(= (path-length START INPUT C-CS2 OUTPUT) 7.345093)
		(= (path-length START INPUT C-DS INPUT) 7.652717)
		(= (path-length START INPUT C-DS OUTPUT) 5.493162)
	)
	(:goal (order-fulfilled o1))
)