(define (problem rcll-production-079-durative)
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
		(= (path-length C-BS INPUT C-BS OUTPUT) 2.763203)
		(= (path-length C-BS INPUT C-CS1 INPUT) 10.243678)
		(= (path-length C-BS INPUT C-CS1 OUTPUT) 9.405694)
		(= (path-length C-BS INPUT C-CS2 INPUT) 3.701392)
		(= (path-length C-BS INPUT C-CS2 OUTPUT) 2.521891)
		(= (path-length C-BS INPUT C-DS INPUT) 7.703308)
		(= (path-length C-BS INPUT C-DS OUTPUT) 7.388236)
		(= (path-length C-BS OUTPUT C-BS INPUT) 2.763202)
		(= (path-length C-BS OUTPUT C-CS1 INPUT) 11.170231)
		(= (path-length C-BS OUTPUT C-CS1 OUTPUT) 10.701686)
		(= (path-length C-BS OUTPUT C-CS2 INPUT) 2.278427)
		(= (path-length C-BS OUTPUT C-CS2 OUTPUT) 3.864765)
		(= (path-length C-BS OUTPUT C-DS INPUT) 6.280342)
		(= (path-length C-BS OUTPUT C-DS OUTPUT) 8.684229)
		(= (path-length C-CS1 INPUT C-BS INPUT) 10.243678)
		(= (path-length C-CS1 INPUT C-BS OUTPUT) 11.170233)
		(= (path-length C-CS1 INPUT C-CS1 OUTPUT) 3.581078)
		(= (path-length C-CS1 INPUT C-CS2 INPUT) 9.461175)
		(= (path-length C-CS1 INPUT C-CS2 OUTPUT) 7.908775)
		(= (path-length C-CS1 INPUT C-DS INPUT) 5.020761)
		(= (path-length C-CS1 INPUT C-DS OUTPUT) 4.729339)
		(= (path-length C-CS1 OUTPUT C-BS INPUT) 9.405693)
		(= (path-length C-CS1 OUTPUT C-BS OUTPUT) 10.701687)
		(= (path-length C-CS1 OUTPUT C-CS1 INPUT) 3.581078)
		(= (path-length C-CS1 OUTPUT C-CS2 INPUT) 8.992629)
		(= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 7.07079)
		(= (path-length C-CS1 OUTPUT C-DS INPUT) 7.562325)
		(= (path-length C-CS1 OUTPUT C-DS OUTPUT) 6.521162)
		(= (path-length C-CS2 INPUT C-BS INPUT) 3.701392)
		(= (path-length C-CS2 INPUT C-BS OUTPUT) 2.278428)
		(= (path-length C-CS2 INPUT C-CS1 INPUT) 9.461175)
		(= (path-length C-CS2 INPUT C-CS1 OUTPUT) 8.992629)
		(= (path-length C-CS2 INPUT C-CS2 OUTPUT) 3.613353)
		(= (path-length C-CS2 INPUT C-DS INPUT) 4.571285)
		(= (path-length C-CS2 INPUT C-DS OUTPUT) 6.975172)
		(= (path-length C-CS2 OUTPUT C-BS INPUT) 2.521891)
		(= (path-length C-CS2 OUTPUT C-BS OUTPUT) 3.864765)
		(= (path-length C-CS2 OUTPUT C-CS1 INPUT) 7.908775)
		(= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 7.07079)
		(= (path-length C-CS2 OUTPUT C-CS2 INPUT) 3.613353)
		(= (path-length C-CS2 OUTPUT C-DS INPUT) 5.747776)
		(= (path-length C-CS2 OUTPUT C-DS OUTPUT) 5.053333)
		(= (path-length C-DS INPUT C-BS INPUT) 7.703308)
		(= (path-length C-DS INPUT C-BS OUTPUT) 6.280342)
		(= (path-length C-DS INPUT C-CS1 INPUT) 5.020761)
		(= (path-length C-DS INPUT C-CS1 OUTPUT) 7.562325)
		(= (path-length C-DS INPUT C-CS2 INPUT) 4.571285)
		(= (path-length C-DS INPUT C-CS2 OUTPUT) 5.747777)
		(= (path-length C-DS INPUT C-DS OUTPUT) 3.15937)
		(= (path-length C-DS OUTPUT C-BS INPUT) 7.388237)
		(= (path-length C-DS OUTPUT C-BS OUTPUT) 8.684229)
		(= (path-length C-DS OUTPUT C-CS1 INPUT) 4.729339)
		(= (path-length C-DS OUTPUT C-CS1 OUTPUT) 6.521162)
		(= (path-length C-DS OUTPUT C-CS2 INPUT) 6.975172)
		(= (path-length C-DS OUTPUT C-CS2 OUTPUT) 5.053333)
		(= (path-length C-DS OUTPUT C-DS INPUT) 3.15937)
		(= (path-length START INPUT C-BS INPUT) 1.312582)
		(= (path-length START INPUT C-BS OUTPUT) 3.385028)
		(= (path-length START INPUT C-CS1 INPUT) 9.341178)
		(= (path-length START INPUT C-CS1 OUTPUT) 8.503193)
		(= (path-length START INPUT C-CS2 INPUT) 3.133616)
		(= (path-length START INPUT C-CS2 OUTPUT) 1.954115)
		(= (path-length START INPUT C-DS INPUT) 7.135531)
		(= (path-length START INPUT C-DS OUTPUT) 6.485734)
	)
	(:goal (order-fulfilled o1))
)