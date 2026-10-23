# Mark Chmielecki - Final Exam Problem 2

class SecurityKB:
    def __init__(self, initial_facts):
        # The 'Model' of the world: a set of things we know to be true
        self.facts = set(initial_facts)
        # PART 1: Define the Knowledge Base
        # Format: ( [List of Prerequisites], Result_to_Add )

        self.rules = [(['has_keycard', 'biometric_passed'], 'access_allowed'),    # Rule: If has_keycard and biometric_passed, then access_allowed          
                    (['system_armed', 'motion_detected'], 'alarm'),               # Add the 'Alarm' rule here
                    (['staff_member', 'high_clearance'], 'access_allowed'),       # Rule 2: staff_member AND high_clearance->access_allowed
                    (['access_allowed', 'system_disarmed'],'unlock_door')]        # Rule: If access_allowed and system_disarmed, then unlock_door

    def infer(self):
        """
        PART 2: Implement Forward Chaining.
        Requirement: The loop must continue until the Knowledge Base stops growing.
        """
        changed = True

        while changed:
            changed = False # assume no new facts will be added

            for prerequisites, result in self.rules: #checks each of the rules for fact, if new facts are found than they are added to the knowledge based

                if all(fact in self.facts for fact in prerequisites):

                    if result not in self.facts:
                        self.facts.add(result)
                        changed = True

# --- TEST CASE ---
#Scenario: A staff member with high clearance enters while the system is disarmed.
initial_info = ['staff_member', 'high_clearance', 'system_disarmed']
agent_kb = SecurityKB(initial_info)
agent_kb.infer()
print(f"Final Facts in KB: {agent_kb.facts}")
print(f"Action - Unlock Door: {'unlock_door' in agent_kb.facts}")