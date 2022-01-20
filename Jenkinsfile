pipeline {
  options {
    disableConcurrentBuilds()
  }
  agent any
  stages {
    stage('Execute CI') {
      steps {
        sh '/usr/local/bin/fab -f /opt/jenkins/jenkins_scripts/timemachine_ci_fab.py ci:'+env.WORKSPACE+','+env.GIT_COMMIT
      }
    }

  }
}
