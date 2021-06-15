(define (problem rcll-production-026-durative)
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
		(order-gate o1 GATE-3)
		(= (path-length C-BS INPUT C-BS OUTPUT) 2.511727)
		(= (path-length C-BS INPUT C-CS1 INPUT) 10.130019)
		(= (path-length C-BS INPUT C-CS1 OUTPUT) 7.081334)
		(= (path-length C-BS INPUT C-CS2 INPUT) 3.23261)
		(= (path-length C-BS INPUT C-CS2 OUTPUT) 4.069756)
		(= (path-length C-BS INPUT C-DS INPUT) 7.658762)
		(= (path-length C-BS INPUT C-DS OUTPUT) 8.080629)
		(= (path-length C-BS OUTPUT C-BS INPUT) 2.511727)
		(= (path-length C-BS OUTPUT C-CS1 INPUT) 12.368324)
		(= (path-length C-BS OUTPUT C-CS1 OUTPUT) 9.319639)
		(= (path-length C-BS OUTPUT C-CS2 INPUT) 5.470916)
		(= (path-length C-BS OUTPUT C-CS2 OUTPUT) 6.308062)
		(= (path-length C-BS OUTPUT C-DS INPUT) 6.100646)
		(= (path-length C-BS OUTPUT C-DS OUTPUT) 6.496589)
		(= (path-length C-CS1 INPUT C-BS INPUT) 10.130018)
		(= (path-length C-CS1 INPUT C-BS OUTPUT) 12.368325)
		(= (path-length C-CS1 INPUT C-CS1 OUTPUT) 4.137381)
		(= (path-length C-CS1 INPUT C-CS2 INPUT) 7.105592)
		(= (path-length C-CS1 INPUT C-CS2 OUTPUT) 6.621799)
		(= (path-length C-CS1 INPUT C-DS INPUT) 8.209899)
		(= (path-length C-CS1 INPUT C-DS OUTPUT) 8.304967)
		(= (path-length C-CS1 OUTPUT C-BS INPUT) 7.081334)
		(= (path-length C-CS1 OUTPUT C-BS OUTPUT) 9.319641)
		(= (path-length C-CS1 OUTPUT C-CS1 INPUT) 4.137381)
		(= (path-length C-CS1 OUTPUT C-CS2 INPUT) 4.056907)
		(= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 4.198783)
		(= (path-length C-CS1 OUTPUT C-DS INPUT) 6.009479)
		(= (path-length C-CS1 OUTPUT C-DS OUTPUT) 6.669363)
		(= (path-length C-CS2 INPUT C-BS INPUT) 3.23261)
		(= (path-length C-CS2 INPUT C-BS OUTPUT) 5.470915)
		(= (path-length C-CS2 INPUT C-CS1 INPUT) 7.105592)
		(= (path-length C-CS2 INPUT C-CS1 OUTPUT) 4.056907)
		(= (path-length C-CS2 INPUT C-CS2 OUTPUT) 2.095082)
		(= (path-length C-CS2 INPUT C-DS INPUT) 5.391837)
		(= (path-length C-CS2 INPUT C-DS OUTPUT) 6.051721)
		(= (path-length C-CS2 OUTPUT C-BS INPUT) 4.069756)
		(= (path-length C-CS2 OUTPUT C-BS OUTPUT) 6.308062)
		(= (path-length C-CS2 OUTPUT C-CS1 INPUT) 6.621799)
		(= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 4.198783)
		(= (path-length C-CS2 OUTPUT C-CS2 INPUT) 2.095082)
		(= (path-length C-CS2 OUTPUT C-DS INPUT) 5.533713)
		(= (path-length C-CS2 OUTPUT C-DS OUTPUT) 6.193597)
		(= (path-length C-DS INPUT C-BS INPUT) 7.658763)
		(= (path-length C-DS INPUT C-BS OUTPUT) 6.100646)
		(= (path-length C-DS INPUT C-CS1 INPUT) 8.209899)
		(= (path-length C-DS INPUT C-CS1 OUTPUT) 6.00948)
		(= (path-length C-DS INPUT C-CS2 INPUT) 5.391837)
		(= (path-length C-DS INPUT C-CS2 OUTPUT) 5.533713)
		(= (path-length C-DS INPUT C-DS OUTPUT) 3.266794)
		(= (path-length C-DS OUTPUT C-BS INPUT) 8.080631)
		(= (path-length C-DS OUTPUT C-BS OUTPUT) 6.496589)
		(= (path-length C-DS OUTPUT C-CS1 INPUT) 8.304967)
		(= (path-length C-DS OUTPUT C-CS1 OUTPUT) 6.669364)
		(= (path-length C-DS OUTPUT C-CS2 INPUT) 6.051722)
		(= (path-length C-DS OUTPUT C-CS2 OUTPUT) 6.193598)
		(= (path-length C-DS OUTPUT C-DS INPUT) 3.266794)
		(= (path-length START INPUT C-BS INPUT) 1.343752)
		(= (path-length START INPUT C-BS OUTPUT) 3.582057)
		(= (path-length START INPUT C-CS1 INPUT) 9.480612)
		(= (path-length START INPUT C-CS1 OUTPUT) 6.431927)
		(= (path-length START INPUT C-CS2 INPUT) 2.583203)
		(= (path-length START INPUT C-CS2 OUTPUT) 3.420349)
		(= (path-length START INPUT C-DS INPUT) 7.009356)
		(= (path-length START INPUT C-DS OUTPUT) 7.431222)
	)
	(:goal (order-fulfilled o1))
)