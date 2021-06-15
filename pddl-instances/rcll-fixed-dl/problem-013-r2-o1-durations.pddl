(define (problem rcll-production-013-durative)
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
		(= (path-length C-BS INPUT C-BS OUTPUT) 3.026292)
		(= (path-length C-BS INPUT C-CS1 INPUT) 10.279427)
		(= (path-length C-BS INPUT C-CS1 OUTPUT) 11.667335)
		(= (path-length C-BS INPUT C-CS2 INPUT) 4.010736)
		(= (path-length C-BS INPUT C-CS2 OUTPUT) 3.034101)
		(= (path-length C-BS INPUT C-DS INPUT) 5.557915)
		(= (path-length C-BS INPUT C-DS OUTPUT) 8.556118)
		(= (path-length C-BS OUTPUT C-BS INPUT) 3.026292)
		(= (path-length C-BS OUTPUT C-CS1 INPUT) 8.431274)
		(= (path-length C-BS OUTPUT C-CS1 OUTPUT) 10.172664)
		(= (path-length C-BS OUTPUT C-CS2 INPUT) 1.840341)
		(= (path-length C-BS OUTPUT C-CS2 OUTPUT) 3.912664)
		(= (path-length C-BS OUTPUT C-DS INPUT) 5.13813)
		(= (path-length C-BS OUTPUT C-DS OUTPUT) 7.450758)
		(= (path-length C-CS1 INPUT C-BS INPUT) 10.279427)
		(= (path-length C-CS1 INPUT C-BS OUTPUT) 8.431275)
		(= (path-length C-CS1 INPUT C-CS1 OUTPUT) 3.647278)
		(= (path-length C-CS1 INPUT C-CS2 INPUT) 7.247489)
		(= (path-length C-CS1 INPUT C-CS2 OUTPUT) 8.067433)
		(= (path-length C-CS1 INPUT C-DS INPUT) 7.476085)
		(= (path-length C-CS1 INPUT C-DS OUTPUT) 5.850607)
		(= (path-length C-CS1 OUTPUT C-BS INPUT) 11.667334)
		(= (path-length C-CS1 OUTPUT C-BS OUTPUT) 10.172665)
		(= (path-length C-CS1 OUTPUT C-CS1 INPUT) 3.647278)
		(= (path-length C-CS1 OUTPUT C-CS2 INPUT) 8.988877)
		(= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 9.45534)
		(= (path-length C-CS1 OUTPUT C-DS INPUT) 6.54842)
		(= (path-length C-CS1 OUTPUT C-DS OUTPUT) 4.137514)
		(= (path-length C-CS2 INPUT C-BS INPUT) 4.010736)
		(= (path-length C-CS2 INPUT C-BS OUTPUT) 1.840341)
		(= (path-length C-CS2 INPUT C-CS1 INPUT) 7.247489)
		(= (path-length C-CS2 INPUT C-CS1 OUTPUT) 8.988877)
		(= (path-length C-CS2 INPUT C-CS2 OUTPUT) 4.162028)
		(= (path-length C-CS2 INPUT C-DS INPUT) 5.387495)
		(= (path-length C-CS2 INPUT C-DS OUTPUT) 6.266973)
		(= (path-length C-CS2 OUTPUT C-BS INPUT) 3.034101)
		(= (path-length C-CS2 OUTPUT C-BS OUTPUT) 3.912664)
		(= (path-length C-CS2 OUTPUT C-CS1 INPUT) 8.067433)
		(= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 9.45534)
		(= (path-length C-CS2 OUTPUT C-CS2 INPUT) 4.162028)
		(= (path-length C-CS2 OUTPUT C-DS INPUT) 3.345922)
		(= (path-length C-CS2 OUTPUT C-DS OUTPUT) 6.344125)
		(= (path-length C-DS INPUT C-BS INPUT) 5.557915)
		(= (path-length C-DS INPUT C-BS OUTPUT) 5.13813)
		(= (path-length C-DS INPUT C-CS1 INPUT) 7.476085)
		(= (path-length C-DS INPUT C-CS1 OUTPUT) 6.548421)
		(= (path-length C-DS INPUT C-CS2 INPUT) 5.387494)
		(= (path-length C-DS INPUT C-CS2 OUTPUT) 3.345922)
		(= (path-length C-DS INPUT C-DS OUTPUT) 3.437204)
		(= (path-length C-DS OUTPUT C-BS INPUT) 8.556118)
		(= (path-length C-DS OUTPUT C-BS OUTPUT) 7.450759)
		(= (path-length C-DS OUTPUT C-CS1 INPUT) 5.850607)
		(= (path-length C-DS OUTPUT C-CS1 OUTPUT) 4.137514)
		(= (path-length C-DS OUTPUT C-CS2 INPUT) 6.266973)
		(= (path-length C-DS OUTPUT C-CS2 OUTPUT) 6.344125)
		(= (path-length C-DS OUTPUT C-DS INPUT) 3.437204)
		(= (path-length START INPUT C-BS INPUT) 3.501697)
		(= (path-length START INPUT C-BS OUTPUT) 1.331302)
		(= (path-length START INPUT C-CS1 INPUT) 7.803566)
		(= (path-length START INPUT C-CS1 OUTPUT) 9.544954)
		(= (path-length START INPUT C-CS2 INPUT) 1.212632)
		(= (path-length START INPUT C-CS2 OUTPUT) 3.652989)
		(= (path-length START INPUT C-DS INPUT) 4.878455)
		(= (path-length START INPUT C-DS OUTPUT) 6.823049)
	)
	(:goal (order-fulfilled o1))
)