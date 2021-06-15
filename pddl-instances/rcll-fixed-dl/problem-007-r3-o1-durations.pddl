(define (problem rcll-production-007-durative)
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
		(= (path-length C-BS INPUT C-BS OUTPUT) 2.611762)
		(= (path-length C-BS INPUT C-CS1 INPUT) 2.454504)
		(= (path-length C-BS INPUT C-CS1 OUTPUT) 3.282615)
		(= (path-length C-BS INPUT C-CS2 INPUT) 7.7869)
		(= (path-length C-BS INPUT C-CS2 OUTPUT) 5.288546)
		(= (path-length C-BS INPUT C-DS INPUT) 4.990722)
		(= (path-length C-BS INPUT C-DS OUTPUT) 7.508314)
		(= (path-length C-BS OUTPUT C-BS INPUT) 2.611762)
		(= (path-length C-BS OUTPUT C-CS1 INPUT) 4.240102)
		(= (path-length C-BS OUTPUT C-CS1 OUTPUT) 4.816305)
		(= (path-length C-BS OUTPUT C-CS2 INPUT) 9.320588)
		(= (path-length C-BS OUTPUT C-CS2 OUTPUT) 7.074143)
		(= (path-length C-BS OUTPUT C-DS INPUT) 6.524411)
		(= (path-length C-BS OUTPUT C-DS OUTPUT) 8.991032)
		(= (path-length C-CS1 INPUT C-BS INPUT) 2.454504)
		(= (path-length C-CS1 INPUT C-BS OUTPUT) 4.240102)
		(= (path-length C-CS1 INPUT C-CS1 OUTPUT) 3.053139)
		(= (path-length C-CS1 INPUT C-CS2 INPUT) 7.45641)
		(= (path-length C-CS1 INPUT C-CS2 OUTPUT) 3.816349)
		(= (path-length C-CS1 INPUT C-DS INPUT) 5.64672)
		(= (path-length C-CS1 INPUT C-DS OUTPUT) 7.193744)
		(= (path-length C-CS1 OUTPUT C-BS INPUT) 3.282615)
		(= (path-length C-CS1 OUTPUT C-BS OUTPUT) 4.816305)
		(= (path-length C-CS1 OUTPUT C-CS1 INPUT) 3.053139)
		(= (path-length C-CS1 OUTPUT C-CS2 INPUT) 6.277532)
		(= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 4.066274)
		(= (path-length C-CS1 OUTPUT C-DS INPUT) 3.481354)
		(= (path-length C-CS1 OUTPUT C-DS OUTPUT) 5.998946)
		(= (path-length C-CS2 INPUT C-BS INPUT) 7.7869)
		(= (path-length C-CS2 INPUT C-BS OUTPUT) 9.32059)
		(= (path-length C-CS2 INPUT C-CS1 INPUT) 7.456411)
		(= (path-length C-CS2 INPUT C-CS1 OUTPUT) 6.277532)
		(= (path-length C-CS2 INPUT C-CS2 OUTPUT) 4.380938)
		(= (path-length C-CS2 INPUT C-DS INPUT) 5.290924)
		(= (path-length C-CS2 INPUT C-DS OUTPUT) 4.528875)
		(= (path-length C-CS2 OUTPUT C-BS INPUT) 5.288545)
		(= (path-length C-CS2 OUTPUT C-BS OUTPUT) 7.074142)
		(= (path-length C-CS2 OUTPUT C-CS1 INPUT) 3.816349)
		(= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 4.066274)
		(= (path-length C-CS2 OUTPUT C-CS2 INPUT) 4.380938)
		(= (path-length C-CS2 OUTPUT C-DS INPUT) 6.004088)
		(= (path-length C-CS2 OUTPUT C-DS OUTPUT) 5.427404)
		(= (path-length C-DS INPUT C-BS INPUT) 4.990722)
		(= (path-length C-DS INPUT C-BS OUTPUT) 6.524412)
		(= (path-length C-DS INPUT C-CS1 INPUT) 5.64672)
		(= (path-length C-DS INPUT C-CS1 OUTPUT) 3.481354)
		(= (path-length C-DS INPUT C-CS2 INPUT) 5.290923)
		(= (path-length C-DS INPUT C-CS2 OUTPUT) 6.004089)
		(= (path-length C-DS INPUT C-DS OUTPUT) 3.652457)
		(= (path-length C-DS OUTPUT C-BS INPUT) 7.508314)
		(= (path-length C-DS OUTPUT C-BS OUTPUT) 8.991033)
		(= (path-length C-DS OUTPUT C-CS1 INPUT) 7.193745)
		(= (path-length C-DS OUTPUT C-CS1 OUTPUT) 5.998946)
		(= (path-length C-DS OUTPUT C-CS2 INPUT) 4.528875)
		(= (path-length C-DS OUTPUT C-CS2 OUTPUT) 5.427405)
		(= (path-length C-DS OUTPUT C-DS INPUT) 3.652456)
		(= (path-length START INPUT C-BS INPUT) 0.782594)
		(= (path-length START INPUT C-BS OUTPUT) 2.568192)
		(= (path-length START INPUT C-CS1 INPUT) 1.729188)
		(= (path-length START INPUT C-CS1 OUTPUT) 3.596919)
		(= (path-length START INPUT C-CS2 INPUT) 8.101204)
		(= (path-length START INPUT C-CS2 OUTPUT) 4.563229)
		(= (path-length START INPUT C-DS INPUT) 5.305026)
		(= (path-length START INPUT C-DS OUTPUT) 7.822618)
	)
	(:goal (order-fulfilled o1))
)