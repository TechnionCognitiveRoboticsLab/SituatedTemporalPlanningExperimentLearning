(define (problem rcll-production-069-durative)
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
		(order-base-color o1 BASE_SILVER)
		(order-cap-color o1 CAP_BLACK)
		(order-gate o1 GATE-1)
		(= (path-length C-BS INPUT C-BS OUTPUT) 2.379678)
		(= (path-length C-BS INPUT C-CS1 INPUT) 4.908973)
		(= (path-length C-BS INPUT C-CS1 OUTPUT) 4.728925)
		(= (path-length C-BS INPUT C-CS2 INPUT) 10.060129)
		(= (path-length C-BS INPUT C-CS2 OUTPUT) 11.284873)
		(= (path-length C-BS INPUT C-DS INPUT) 6.159839)
		(= (path-length C-BS INPUT C-DS OUTPUT) 5.776354)
		(= (path-length C-BS OUTPUT C-BS INPUT) 2.379678)
		(= (path-length C-BS OUTPUT C-CS1 INPUT) 6.45234)
		(= (path-length C-BS OUTPUT C-CS1 OUTPUT) 6.272291)
		(= (path-length C-BS OUTPUT C-CS2 INPUT) 11.603495)
		(= (path-length C-BS OUTPUT C-CS2 OUTPUT) 12.828238)
		(= (path-length C-BS OUTPUT C-DS INPUT) 6.940067)
		(= (path-length C-BS OUTPUT C-DS OUTPUT) 7.31972)
		(= (path-length C-CS1 INPUT C-BS INPUT) 4.908973)
		(= (path-length C-CS1 INPUT C-BS OUTPUT) 6.452339)
		(= (path-length C-CS1 INPUT C-CS1 OUTPUT) 3.502791)
		(= (path-length C-CS1 INPUT C-CS2 INPUT) 6.379979)
		(= (path-length C-CS1 INPUT C-CS2 OUTPUT) 7.604722)
		(= (path-length C-CS1 INPUT C-DS INPUT) 4.632141)
		(= (path-length C-CS1 INPUT C-DS OUTPUT) 2.556366)
		(= (path-length C-CS1 OUTPUT C-BS INPUT) 4.728925)
		(= (path-length C-CS1 OUTPUT C-BS OUTPUT) 6.272291)
		(= (path-length C-CS1 OUTPUT C-CS1 INPUT) 3.502791)
		(= (path-length C-CS1 OUTPUT C-CS2 INPUT) 7.081163)
		(= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 8.340283)
		(= (path-length C-CS1 OUTPUT C-DS INPUT) 2.332824)
		(= (path-length C-CS1 OUTPUT C-DS OUTPUT) 1.119554)
		(= (path-length C-CS2 INPUT C-BS INPUT) 10.060129)
		(= (path-length C-CS2 INPUT C-BS OUTPUT) 11.603496)
		(= (path-length C-CS2 INPUT C-CS1 INPUT) 6.379979)
		(= (path-length C-CS2 INPUT C-CS1 OUTPUT) 7.081164)
		(= (path-length C-CS2 INPUT C-CS2 OUTPUT) 3.838387)
		(= (path-length C-CS2 INPUT C-DS INPUT) 7.119454)
		(= (path-length C-CS2 INPUT C-DS OUTPUT) 6.022017)
		(= (path-length C-CS2 OUTPUT C-BS INPUT) 11.284872)
		(= (path-length C-CS2 OUTPUT C-BS OUTPUT) 12.828238)
		(= (path-length C-CS2 OUTPUT C-CS1 INPUT) 7.604722)
		(= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 8.340283)
		(= (path-length C-CS2 OUTPUT C-CS2 INPUT) 3.838387)
		(= (path-length C-CS2 OUTPUT C-DS INPUT) 8.378574)
		(= (path-length C-CS2 OUTPUT C-DS OUTPUT) 7.281137)
		(= (path-length C-DS INPUT C-BS INPUT) 6.159839)
		(= (path-length C-DS INPUT C-BS OUTPUT) 6.940066)
		(= (path-length C-DS INPUT C-CS1 INPUT) 4.632141)
		(= (path-length C-DS INPUT C-CS1 OUTPUT) 2.332824)
		(= (path-length C-DS INPUT C-CS2 INPUT) 7.119454)
		(= (path-length C-DS INPUT C-CS2 OUTPUT) 8.378574)
		(= (path-length C-DS INPUT C-DS OUTPUT) 3.13835)
		(= (path-length C-DS OUTPUT C-BS INPUT) 5.776354)
		(= (path-length C-DS OUTPUT C-BS OUTPUT) 7.31972)
		(= (path-length C-DS OUTPUT C-CS1 INPUT) 2.556366)
		(= (path-length C-DS OUTPUT C-CS1 OUTPUT) 1.119554)
		(= (path-length C-DS OUTPUT C-CS2 INPUT) 6.022018)
		(= (path-length C-DS OUTPUT C-CS2 OUTPUT) 7.281138)
		(= (path-length C-DS OUTPUT C-DS INPUT) 3.13835)
		(= (path-length START INPUT C-BS INPUT) 1.125739)
		(= (path-length START INPUT C-BS OUTPUT) 3.368991)
		(= (path-length START INPUT C-CS1 INPUT) 4.543069)
		(= (path-length START INPUT C-CS1 OUTPUT) 4.363021)
		(= (path-length START INPUT C-CS2 INPUT) 9.694225)
		(= (path-length START INPUT C-CS2 OUTPUT) 10.918969)
		(= (path-length START INPUT C-DS INPUT) 5.793934)
		(= (path-length START INPUT C-DS OUTPUT) 5.410449)
	)
	(:goal (order-fulfilled o1))
)