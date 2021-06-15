(define (problem rcll-production-043-durative)
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
		(order-gate o1 GATE-3)
		(= (path-length C-BS INPUT C-BS OUTPUT) 2.82946)
		(= (path-length C-BS INPUT C-CS1 INPUT) 8.20183)
		(= (path-length C-BS INPUT C-CS1 OUTPUT) 6.607025)
		(= (path-length C-BS INPUT C-CS2 INPUT) 1.894851)
		(= (path-length C-BS INPUT C-CS2 OUTPUT) 4.079586)
		(= (path-length C-BS INPUT C-DS INPUT) 8.219977)
		(= (path-length C-BS INPUT C-DS OUTPUT) 6.179011)
		(= (path-length C-BS OUTPUT C-BS INPUT) 2.82946)
		(= (path-length C-BS OUTPUT C-CS1 INPUT) 7.028909)
		(= (path-length C-BS OUTPUT C-CS1 OUTPUT) 5.434103)
		(= (path-length C-BS OUTPUT C-CS2 INPUT) 4.395087)
		(= (path-length C-BS OUTPUT C-CS2 OUTPUT) 4.468006)
		(= (path-length C-BS OUTPUT C-DS INPUT) 7.709251)
		(= (path-length C-BS OUTPUT C-DS OUTPUT) 5.006089)
		(= (path-length C-CS1 INPUT C-BS INPUT) 8.201831)
		(= (path-length C-CS1 INPUT C-BS OUTPUT) 7.02891)
		(= (path-length C-CS1 INPUT C-CS1 OUTPUT) 3.00831)
		(= (path-length C-CS1 INPUT C-CS2 INPUT) 8.179565)
		(= (path-length C-CS1 INPUT C-CS2 OUTPUT) 8.252484)
		(= (path-length C-CS1 INPUT C-DS INPUT) 6.580138)
		(= (path-length C-CS1 INPUT C-DS OUTPUT) 6.942951)
		(= (path-length C-CS1 OUTPUT C-BS INPUT) 6.607025)
		(= (path-length C-CS1 OUTPUT C-BS OUTPUT) 5.434103)
		(= (path-length C-CS1 OUTPUT C-CS1 INPUT) 3.00831)
		(= (path-length C-CS1 OUTPUT C-CS2 INPUT) 6.584759)
		(= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 6.657678)
		(= (path-length C-CS1 OUTPUT C-DS INPUT) 4.825294)
		(= (path-length C-CS1 OUTPUT C-DS OUTPUT) 5.188107)
		(= (path-length C-CS2 INPUT C-BS INPUT) 1.894851)
		(= (path-length C-CS2 INPUT C-BS OUTPUT) 4.395087)
		(= (path-length C-CS2 INPUT C-CS1 INPUT) 8.179565)
		(= (path-length C-CS2 INPUT C-CS1 OUTPUT) 6.584759)
		(= (path-length C-CS2 INPUT C-CS2 OUTPUT) 4.05732)
		(= (path-length C-CS2 INPUT C-DS INPUT) 6.981942)
		(= (path-length C-CS2 INPUT C-DS OUTPUT) 6.156745)
		(= (path-length C-CS2 OUTPUT C-BS INPUT) 4.079586)
		(= (path-length C-CS2 OUTPUT C-BS OUTPUT) 4.468006)
		(= (path-length C-CS2 OUTPUT C-CS1 INPUT) 8.252483)
		(= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 6.657678)
		(= (path-length C-CS2 OUTPUT C-CS2 INPUT) 4.05732)
		(= (path-length C-CS2 OUTPUT C-DS INPUT) 4.517404)
		(= (path-length C-CS2 OUTPUT C-DS OUTPUT) 3.734211)
		(= (path-length C-DS INPUT C-BS INPUT) 8.219977)
		(= (path-length C-DS INPUT C-BS OUTPUT) 7.709252)
		(= (path-length C-DS INPUT C-CS1 INPUT) 6.580138)
		(= (path-length C-DS INPUT C-CS1 OUTPUT) 4.825294)
		(= (path-length C-DS INPUT C-CS2 INPUT) 6.981942)
		(= (path-length C-DS INPUT C-CS2 OUTPUT) 4.517404)
		(= (path-length C-DS INPUT C-DS OUTPUT) 3.555409)
		(= (path-length C-DS OUTPUT C-BS INPUT) 6.17901)
		(= (path-length C-DS OUTPUT C-BS OUTPUT) 5.006089)
		(= (path-length C-DS OUTPUT C-CS1 INPUT) 6.94295)
		(= (path-length C-DS OUTPUT C-CS1 OUTPUT) 5.188107)
		(= (path-length C-DS OUTPUT C-CS2 INPUT) 6.156744)
		(= (path-length C-DS OUTPUT C-CS2 OUTPUT) 3.734211)
		(= (path-length C-DS OUTPUT C-DS INPUT) 3.555409)
		(= (path-length START INPUT C-BS INPUT) 3.737139)
		(= (path-length START INPUT C-BS OUTPUT) 1.462494)
		(= (path-length START INPUT C-CS1 INPUT) 6.488777)
		(= (path-length START INPUT C-CS1 OUTPUT) 4.893971)
		(= (path-length START INPUT C-CS2 INPUT) 4.262609)
		(= (path-length START INPUT C-CS2 OUTPUT) 4.335529)
		(= (path-length START INPUT C-DS INPUT) 7.16912)
		(= (path-length START INPUT C-DS OUTPUT) 4.465957)
	)
	(:goal (order-fulfilled o1))
)