# forms.py
from django import forms

class TeamMembersForm(forms.Form):
    number_of_members = forms.IntegerField()

from django import forms

class TeamForm(forms.Form):
    def __init__(self, *args, team_size=1, **kwargs):
        super(TeamForm, self).__init__(*args, **kwargs)
        for i in range(1, team_size+1):
            self.fields[f'team_member_{i}'] = forms.CharField(label=f'Team Member {i}', required=False)