pipeline {
  agent any
  stages {
    stage('Execute CI') {
      steps {
        sh '/usr/local/bin/fab -f /opt/jenkins/jenkins_scripts/jankmachine_ci_fab.py ci:'+env.WORKSPACE+','+env.GIT_COMMIT
      }
    }

  }
}
