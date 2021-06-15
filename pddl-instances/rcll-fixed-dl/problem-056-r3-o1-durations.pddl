(define (problem rcll-production-056-durative)
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
		(order-base-color o1 BASE_BLACK)
		(order-cap-color o1 CAP_BLACK)
		(order-gate o1 GATE-2)
		(= (path-length C-BS INPUT C-BS OUTPUT) 3.060042)
		(= (path-length C-BS INPUT C-CS1 INPUT) 3.135121)
		(= (path-length C-BS INPUT C-CS1 OUTPUT) 3.280492)
		(= (path-length C-BS INPUT C-CS2 INPUT) 7.844877)
		(= (path-length C-BS INPUT C-CS2 OUTPUT) 5.608488)
		(= (path-length C-BS INPUT C-DS INPUT) 7.777063)
		(= (path-length C-BS INPUT C-DS OUTPUT) 8.173609)
		(= (path-length C-BS OUTPUT C-BS INPUT) 3.060042)
		(= (path-length C-BS OUTPUT C-CS1 INPUT) 0.822442)
		(= (path-length C-BS OUTPUT C-CS1 OUTPUT) 2.955628)
		(= (path-length C-BS OUTPUT C-CS2 INPUT) 8.108926)
		(= (path-length C-BS OUTPUT C-CS2 OUTPUT) 5.872535)
		(= (path-length C-BS OUTPUT C-DS INPUT) 7.452199)
		(= (path-length C-BS OUTPUT C-DS OUTPUT) 6.768636)
		(= (path-length C-CS1 INPUT C-BS INPUT) 3.135121)
		(= (path-length C-CS1 INPUT C-BS OUTPUT) 0.822442)
		(= (path-length C-CS1 INPUT C-CS1 OUTPUT) 3.030707)
		(= (path-length C-CS1 INPUT C-CS2 INPUT) 8.184005)
		(= (path-length C-CS1 INPUT C-CS2 OUTPUT) 5.947614)
		(= (path-length C-CS1 INPUT C-DS INPUT) 7.527278)
		(= (path-length C-CS1 INPUT C-DS OUTPUT) 6.770996)
		(= (path-length C-CS1 OUTPUT C-BS INPUT) 3.280492)
		(= (path-length C-CS1 OUTPUT C-BS OUTPUT) 2.955628)
		(= (path-length C-CS1 OUTPUT C-CS1 INPUT) 3.030707)
		(= (path-length C-CS1 OUTPUT C-CS2 INPUT) 5.926275)
		(= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 4.245327)
		(= (path-length C-CS1 OUTPUT C-DS INPUT) 4.82563)
		(= (path-length C-CS1 OUTPUT C-DS OUTPUT) 5.222177)
		(= (path-length C-CS2 INPUT C-BS INPUT) 7.844878)
		(= (path-length C-CS2 INPUT C-BS OUTPUT) 8.108926)
		(= (path-length C-CS2 INPUT C-CS1 INPUT) 8.184004)
		(= (path-length C-CS2 INPUT C-CS1 OUTPUT) 5.926275)
		(= (path-length C-CS2 INPUT C-CS2 OUTPUT) 2.236985)
		(= (path-length C-CS2 INPUT C-DS INPUT) 4.901609)
		(= (path-length C-CS2 INPUT C-DS OUTPUT) 6.542119)
		(= (path-length C-CS2 OUTPUT C-BS INPUT) 5.608488)
		(= (path-length C-CS2 OUTPUT C-BS OUTPUT) 5.872535)
		(= (path-length C-CS2 OUTPUT C-CS1 INPUT) 5.947614)
		(= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 4.245327)
		(= (path-length C-CS2 OUTPUT C-CS2 INPUT) 2.236985)
		(= (path-length C-CS2 OUTPUT C-DS INPUT) 4.309491)
		(= (path-length C-CS2 OUTPUT C-DS OUTPUT) 5.95)
		(= (path-length C-DS INPUT C-BS INPUT) 7.777063)
		(= (path-length C-DS INPUT C-BS OUTPUT) 7.4522)
		(= (path-length C-DS INPUT C-CS1 INPUT) 7.527279)
		(= (path-length C-DS INPUT C-CS1 OUTPUT) 4.82563)
		(= (path-length C-DS INPUT C-CS2 INPUT) 4.90161)
		(= (path-length C-DS INPUT C-CS2 OUTPUT) 4.309491)
		(= (path-length C-DS INPUT C-DS OUTPUT) 3.234657)
		(= (path-length C-DS OUTPUT C-BS INPUT) 8.173611)
		(= (path-length C-DS OUTPUT C-BS OUTPUT) 6.768637)
		(= (path-length C-DS OUTPUT C-CS1 INPUT) 6.770996)
		(= (path-length C-DS OUTPUT C-CS1 OUTPUT) 5.222177)
		(= (path-length C-DS OUTPUT C-CS2 INPUT) 6.542119)
		(= (path-length C-DS OUTPUT C-CS2 OUTPUT) 5.95)
		(= (path-length C-DS OUTPUT C-DS INPUT) 3.234657)
		(= (path-length START INPUT C-BS INPUT) 1.760866)
		(= (path-length START INPUT C-BS OUTPUT) 2.024913)
		(= (path-length START INPUT C-CS1 INPUT) 2.099992)
		(= (path-length START INPUT C-CS1 OUTPUT) 2.245363)
		(= (path-length START INPUT C-CS2 INPUT) 6.140415)
		(= (path-length START INPUT C-CS2 OUTPUT) 3.904025)
		(= (path-length START INPUT C-DS INPUT) 6.741934)
		(= (path-length START INPUT C-DS OUTPUT) 7.13848)
	)
	(:goal (order-fulfilled o1))
)