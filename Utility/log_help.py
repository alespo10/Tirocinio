log = {
    'helpdesk': {
        'event_attribute': [
            'case', 'activity', 'resource', 'timestamp', 'customer', 'product',
            'responsiblesection', 'seriousness2'
        ],
        'trace_attribute': [],
        'event_template': (
            "the activity {activity} was performed by resource {resource} "
            "at time {timestamp} for customer {customer}, on product {product}, "
            "in the responsible section {responsiblesection}, with a seriousness level of {seriousness2}."
        ),
        'trace_template': '',
        'target': 'activity'
    },

    'sepsis': {
        'event_attribute': [
            'case', 'activity', 'orggroup', 'timestamp', 'Leucocytes', 'CRP', 'LacticAcid'
        ],
        'trace_attribute': ['Age', 'InfectionSuspected', 'DiagnosticBlood', 'DisfuncOrg'],
        'event_template': (
            "the organization {orggroup} carried out the activity {activity} on {timestamp}. During this time, the recorded levels were as follows: Leucocytes at {Leucocytes}, CRP at {CRP}, and Lactic Acid at {LacticAcid}."
        ),
        'trace_template': (
            "The patient, who is {Age} years old, is suspected to have an infection."
            "it's {DiagnosticBlood} that blood diagnostics was made, and it's {DisfuncOrg} that an organ dysfunction are present"

        ),
        'target': 'activity'
    },

    'bpic2020': {
        'event_attribute': [
            'case', 'activity', 'resource', 'timestamp', 'Role'
        ],
        'trace_attribute': ['Org', 'Project', 'Task'],
        'event_template': (
            "the activity {activity} was performed by resource {resource} with role {Role} at time {timestamp}."
        ),
        'trace_template': (
            "{Org} managed the {Project} for task {Task}."
        ),
        'target': 'activity'
    },

    'BPIC15_1': {
        'event_attribute': [
            'case', 'activity', 'resource', 'timestamp', 'question', 'monitoringResource'
        ],
        'trace_attribute': [
            'parts', 'responsibleactor', 'lastphase', 'landregisterid', 'casestatus', 'sumleges'
        ],
        'event_template': (
            "the activity {activity} was performed by resource {resource} "
            "at time {timestamp}. The open question {question} concerned {monitoringResource}."
        ),
        'trace_template': (
            "The application concerned the status {casestatus} of {parts} as part of {lastphase} "
            "in the project associated with Land Register ID: {landregisterid}, with sum of leges {sumleges} "
            "and responsible actor {responsibleactor}."
        ),
        'target': 'activity'
    },

    'bpic2017_o': {
        'event_attribute': [
            'case', 'activity', 'resource', 'timestamp', 'action'
        ],
        'trace_attribute': [
            "MonthlyCost", "CreditScore", "FirstWithdrawalAmount", "OfferedAmount", "NumberOfTerms"
        ],
        'event_template': (
            "the activity {activity} was performed by resource {resource} "
            "with status {action} at time {timestamp}."
        ),
        'trace_template': (
            "The monthly cost {MonthlyCost} for the loan, determined based on the credit score {CreditScore}, "
            "calculated considering the first withdrawal amount {FirstWithdrawalAmount}, "
            "the offered amount {OfferedAmount}, and the number of terms {NumberOfTerms}."
        ),
        'target': 'activity'
    },

    'mip': {
        'event_attribute': [
            'case', 'activity', 'resource', 'timestamp', 'numsession', 'userid', 'turn',
            'userutterance', 'chatbotresponse'
        ],
        'trace_attribute': [],
        'event_template': (
            "the activity {activity} was performed by {resource} "
            "at time {timestamp}. In session {numsession}, turn {turn}, the user's utterance was '{userutterance}' "
            "and the chatbot's response was '{chatbotresponse}'."
        ),
        'trace_template': '',
        'target': 'activity'
    }
}
