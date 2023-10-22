from django.shortcuts import render
from django.http import HttpResponse
from .forms import TeamMembersForm, TeamForm
from django.shortcuts import redirect

# Create your views here.

def index(request):

    # Page from the theme 
    return render(request, 'pages/custom-index.html')

def home(request):

    # Page from the theme 
    return render(request, 'pages/custom-home.html')


def teams(request):

    if request.method == 'POST':
        form = TeamMembersForm(request.POST)
        if form.is_valid():
            number_of_members = form.cleaned_data['number_of_members']
            # redirect to teams 2 which has N fields
            return redirect('teams2', team_size=number_of_members)

    else:
        form = TeamMembersForm()

    return render(request, 'pages/custom-teams.html', {'form': form})

from django.shortcuts import render
from .forms import TeamForm  # Import your TeamForm

def teams2(request, team_size):
    if request.method == 'POST':
        form = TeamForm(request.POST, team_size=team_size)
        if form.is_valid():
            data = form.cleaned_data
            print(data)
            d = {}
            for i in range(1, team_size+1):
                d[f'team_member_{i}'] = data[f'team_member_{i}']
            
            for key, value in d.items():
                value = value.split(';')
                vals = []
                for v in value:
                    if v.strip() != '':
                        vals.append(v.strip())
                d[key] = vals
            print(d)

            OUTPUT = None
            return render(request, 'pages/custom-result.html', {'query': d, 'OUTPUT': OUTPUT})
    else:
        form = TeamForm(team_size=team_size)

    return render(request, 'pages/custom-form.html', {'form': form, 'team_size': team_size, 'field_range': range(1, team_size+1)})

def result(request):
    return render(request, 'pages/custom-result.html')